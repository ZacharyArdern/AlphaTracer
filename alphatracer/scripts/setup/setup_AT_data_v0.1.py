#!/usr/bin/env python3
"""
setup_AT_data_v0.1.py — Build the AlphaFold DB representative set and ESM Atlas non-AFDB database.

DIRECTORY LAYOUT:
  Run this script from inside Data/AT_DBs/. The following sibling directories
  will be created automatically if they do not exist:

    Data/
      AT_DBs/       ← run script here (pwd); all intermediate/processed files
        tmp/        ← created automatically
      AFDB/         ← raw AFDB files (afdb_v6_seqs.fa)
      ESM_Atlas/    ← raw ESM Atlas files (esm_plddt60.parquet, esm_plddt60.fasta)
      UniProt/      ← UniRef50 tarball and uniref50.xml

STAGES:
  download                     Download AFDB sequences, UniRef50 XML, and ESM Atlas
                                  → ../AFDB/afdb_v6_seqs.fa
                                     ../UniProt/uniref50.xml
                                     ../ESM_Atlas/esm_plddt60.parquet + .fasta
  cluster_afdb                 Match UniRef50 cluster reps to AFDB structures
                                  → AFDBv6_uniref50_reps.fasta
                                     uniparc_afdb_clusters.tsv
  prep_compare                UniParc pre-filter + build AFDB diamond DB (run on head node)
                                  → esm_plddt60_no_uniparc.fasta
                                     afdb_uniref50_reps_db.dmnd
  compare_esm_afdb             Diamond search one ESM chunk vs AFDB (run per chunk via bsub loop)
                                  → chunk_hits/<chunk>.tsv
  merge_and_cluster_esmatlas   Merge hits, exclude AFDB-covered seqs, linclust, extract reps
                                  → esm_plddt60_non-afdb_reps.fasta.zst

PREREQUISITES (must be on PATH before running):
  conda   — e.g. "module load conda" or activate your conda installation
  The setup_env stage installs all other tools into the conda env automatically.

USAGE (run from Data/AT_DBs/):
  python setup_AT_data_v0.1.py --stage setup_env
  python setup_AT_data_v0.1.py --stage download --threads 16
  python setup_AT_data_v0.1.py --stage cluster_afdb
  python setup_AT_data_v0.1.py --stage prep_compare
  seqkit split2 -p 50 esm_plddt60_no_uniparc.fasta -O esm_chunks/ --quiet
  ls esm_chunks/*.fasta > chunks.txt
  for i in $(cat chunks.txt); do
    python setup_AT_data_v0.1.py --stage compare_esm_afdb --chunk $i --threads 32  # 48GB mem recommended
  done
  # merge hits + exclude + linclust workers + finalise (see LSF instructions below)

  On HPC systems, submit each stage as a job with appropriate resources, e.g. LSF:
  python setup_AT_data_v0.1.py --stage setup_env
  bsub.py --threads 32 -q normal 64   download.log  "python setup_AT_data_v0.1.py --stage download --threads 32"
  bsub.py --threads 8  -q normal 48   clafdb.log    "python setup_AT_data_v0.1.py --stage cluster_afdb"
  python setup_AT_data_v0.1.py --stage prep_compare                              # head node
  seqkit split2 -p 50 esm_plddt60_no_uniparc.fasta -O esm_chunks/ --quiet       # head node
  ls esm_chunks/*.fasta > chunks.txt
  while read i; do
    bsub.py --threads 32 -q long 48 compare_$(basename $i .fasta).log \
      "python setup_AT_data_v0.1.py --stage compare_esm_afdb --chunk $i --threads 32"
  done < chunks.txt
  # merge hits + exclude AFDB-covered seqs (single job):
  bsub.py --threads 8 -q normal 200 merge_excl.log "python setup_AT_data_v0.1.py --stage merge_and_cluster_esmatlas --threads 8"
  # parallel linclust across 10 jobs (submit after merge_excl completes):
  rm -rf tmp/                                                                     # must remove before each fresh linclust run
  for i in $(seq 1 10); do
    bsub.py --threads 32 -q long 96 linclust_worker_${i}.log \
      "python setup_AT_data_v0.1.py --stage linclust_worker --threads 32"
  done
  # after all linclust workers finish, extract reps + compress:
  bsub.py --threads 32 -q normal 120 merge_clesm.log "python setup_AT_data_v0.1.py --stage merge_and_cluster_esmatlas --threads 32"

REQUIRED PROGRAMS (via at_dbs_setup conda env or system PATH):
  diamond  >= 2.2.4
  seqkit   >= 2.9.0
  wget, pigz, zstd
  Python:  lance, pyarrow

REQUIRED FILES (per stage):
  download:                   internet access to S3 and EBI/UniProt FTP
  cluster_afdb:               ../AFDB/afdb_v6_seqs.fa + ../UniProt/uniref50.xml
  prep_compare:              ../ESM_Atlas/esm_plddt60.fasta + uniparc_afdb_clusters.tsv + AFDBv6_uniref50_reps.fasta
  compare_esm_afdb:           esm_plddt60_no_uniparc.fasta + afdb_uniref50_reps_db.dmnd + chunk fasta
  merge_and_cluster_esmatlas: chunk_hits/*.tsv + esm_plddt60_no_uniparc.fasta
"""

import argparse, math, os, re, subprocess, sys, time, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

WORK_DIR    = Path(".")           # Data/AT_DBs/
AFDB_DIR    = Path("../AFDB")
ESM_DIR     = Path("../ESM_Atlas")
UNIPROT_DIR = Path("../UniProt")
TMP_DIR     = Path("./tmp")
PTMP2_DIR   = Path("./tmp2")

# ── Parameters ────────────────────────────────────────────────────────────────

ESM_S3       = "s3://esm-protein-atlas/v1/folds/folds_1B.lance"
PLDDT_MIN    = 0.60
N_CHUNKS     = 50
APPROX_ID    = 50
MEMBER_COVER = 90

AFDB_URL     = "https://ftp.ebi.ac.uk/pub/databases/alphafold/sequences.fasta"
# release 2025_03 matches AlphaFold DB v6; tarball contains nested uniref50.tar
UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2025_03/uniref/uniref2025_03.tar.gz"

# ── Utilities ─────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(cmd, label=""):
    if label: log(label)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True)

def exists(path, label):
    if Path(path).exists():
        log(f"Skipping {label} — {path} exists.")
        return True
    return False

# ── Stage 0: setup_env ───────────────────────────────────────────────────────

def stage_setup_env():
    log("=== setup_env ===")
    ENV = "at_dbs_setup"
    DIAMOND_MIN = (2, 2, 4)   # linclust requires >= 2.2.4
    # diamond always installed via bioconda to enforce minimum version
    CONDA_PKGS  = ["python", "pyarrow", "diamond>=2.2.4"]
    BINARY_PKGS = ["seqkit", "pigz", "zstd", "wget"]

    missing = [p for p in BINARY_PKGS
               if subprocess.run(f"command -v {p}", shell=True, capture_output=True).returncode != 0]
    # check diamond version separately (must be >= 2.2.4 for linclust)
    dmnd = subprocess.run("diamond version", shell=True, capture_output=True, text=True)
    dmnd_ok = False
    if dmnd.returncode == 0:
        import re as _re
        m = _re.search(r"(\d+)\.(\d+)\.(\d+)", dmnd.stdout)
        if m and tuple(int(x) for x in m.groups()) >= DIAMOND_MIN:
            dmnd_ok = True
    if not dmnd_ok:
        log(f"diamond on PATH is absent or < {'.'.join(str(x) for x in DIAMOND_MIN)} — will install in conda env")

    log(f"Missing from PATH: {missing if missing else 'none'}")

    # prefer conda; use mamba only if conda is absent
    installer = "conda" if subprocess.run("command -v conda", shell=True, capture_output=True).returncode == 0 else "mamba"

    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    if any(line.startswith(ENV) for line in result.stdout.splitlines()):
        log(f"Conda env '{ENV}' already exists — checking installed packages...")
        # check which binaries are actually missing from the env
        missing_in_env = [p for p in BINARY_PKGS
                          if subprocess.run(f"conda run -n {ENV} command -v {p}",
                                            shell=True, capture_output=True).returncode != 0]
        # check diamond version inside env
        dmnd_env = subprocess.run(f"conda run -n {ENV} diamond version",
                                  shell=True, capture_output=True, text=True)
        dmnd_env_ok = False
        if dmnd_env.returncode == 0:
            m = re.search(r"(\d+)\.(\d+)\.(\d+)", dmnd_env.stdout)
            if m and tuple(int(x) for x in m.groups()) >= DIAMOND_MIN:
                dmnd_env_ok = True
        pkgs_to_add = missing_in_env + ([] if dmnd_env_ok else ["diamond>=2.2.4"])
        if pkgs_to_add:
            log(f"Installing missing packages into env: {pkgs_to_add}")
            run([installer, "install", "-n", ENV, "-y", "-c", "conda-forge", "-c", "bioconda"] + pkgs_to_add)
        else:
            log("All packages already present in env — nothing to install")
    else:
        pkgs = CONDA_PKGS + missing
        log(f"Creating conda env '{ENV}' with: {pkgs} (using {installer})")
        run([installer, "create", "-n", ENV, "-y", "-c", "conda-forge", "-c", "bioconda"] + pkgs)
        run(f"conda run -n {ENV} pip install pylance", "Installing lance via pip...")
        log(f"Done. Run stages with: python setup_AT_data_v0.1.py --stage ...")

# ── Stage 1: download ─────────────────────────────────────────────────────────

def stage_download(threads):
    log("=== download ===")
    for d in (AFDB_DIR, ESM_DIR, UNIPROT_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    afdb_fa      = AFDB_DIR    / "afdb_v6_seqs.fa"
    uniref_tar   = UNIPROT_DIR / "uniref2025_03.tar.gz"
    uniref50_tar = UNIPROT_DIR / "uniref50.tar"
    uniref_xml   = UNIPROT_DIR / "uniref50.xml"
    pq_out       = ESM_DIR     / "esm_plddt60.parquet"
    fa_out       = ESM_DIR     / "esm_plddt60.fasta"

    # AFDB sequences
    if not exists(afdb_fa, "AFDB sequences"):
        run(f"wget -c -O {afdb_fa} {AFDB_URL}", "Downloading AFDB sequences (~110 GB)...")

    # UniRef50 XML
    if not uniref_xml.exists() and not uniref_tar.exists():
        run(f"wget -c -P {UNIPROT_DIR} {UNIREF50_URL}",
            "Downloading UniRef50 tarball (release 2025_03)...")
    if not uniref_xml.exists() and uniref_tar.exists():
        if not uniref50_tar.exists():
            run(f"tar -I pigz -xf {uniref_tar} -C {UNIPROT_DIR} uniref50.tar",
                "Extracting uniref50.tar from outer tarball (pigz)...")
        run(f"tar -xf {uniref50_tar} -C {UNIPROT_DIR} uniref50.xml",
            "Extracting uniref50.xml from uniref50.tar...")

    # ESM Atlas parquet
    if not exists(pq_out, "ESM Atlas download"):
        import lance, pyarrow.parquet as pq
        ds = lance.dataset(ESM_S3, storage_options={"aws_skip_signature": "true"})
        frags = list(ds.get_fragments())
        log(f"ESM Atlas: {len(frags):,} fragments; streaming pLDDT >= {PLDDT_MIN:.0%} with {threads} threads...")
        fetch = lambda frag: frag.to_table(columns=["header","sequence","mean_plddt","per_residue_plddt"],
                                            filter=f"mean_plddt >= {PLDDT_MIN}")
        t0, writer, written = time.perf_counter(), None, 0
        with ThreadPoolExecutor(max_workers=min(threads, 16)) as pool:
            for i, tbl in enumerate(pool.map(fetch, frags)):
                if len(tbl) == 0: continue
                if writer is None: writer = pq.ParquetWriter(str(pq_out), tbl.schema)
                writer.write_table(tbl); written += len(tbl)
                if (i+1) % 1000 == 0: log(f"  {i+1}/{len(frags)} fragments — {written:,} records")
        if writer: writer.close()
        log(f"Done: {written:,} records in {(time.perf_counter()-t0)/3600:.1f}h")

    # ESM Atlas fasta — independent per-chunk reads to keep memory bounded
    RGS_PER_CHUNK = 5000  # ~30M rows per chunk (109554 row groups / ~5000 each)
    if not exists(fa_out, "parquet->fasta"):
        import gc, math, pyarrow as pa, pyarrow.parquet as pq
        log("Validating parquet footer...")
        try:
            meta = pq.read_metadata(pq_out)
            n_rg = meta.num_row_groups
            log(f"  Parquet OK: {meta.num_rows:,} rows, {n_rg:,} row groups")
        except Exception as e:
            raise RuntimeError(f"Parquet file appears incomplete or corrupt: {e}")
        n_chunks = math.ceil(n_rg / RGS_PER_CHUNK)
        t0 = time.time()
        for ci in range(n_chunks):
            chunk_path = ESM_DIR / f"esm_chunk_{ci:04d}.fasta"
            if chunk_path.exists():
                log(f"  Skipping existing chunk {ci+1}/{n_chunks}")
                continue
            rgs = list(range(ci * RGS_PER_CHUNK, min((ci+1) * RGS_PER_CHUNK, n_rg)))
            n = 0
            pf = pq.ParquetFile(pq_out)
            with open(chunk_path, "w") as out:
                for batch in pf.iter_batches(100_000, columns=["header","sequence"],
                                             row_groups=rgs, use_threads=False):
                    h, s = batch.column("header").to_pylist(), batch.column("sequence").to_pylist()
                    out.write("".join(f">{a}\n{b}\n" for a, b in zip(h, s)))
                    n += len(batch)
            del pf; gc.collect()
            log(f"  chunk {ci+1}/{n_chunks} done — {n:,} rows ({time.time()-t0:.0f}s)")
        log("Concatenating chunks...")
        run(f"cat {' '.join(str(p) for p in sorted(ESM_DIR.glob('esm_chunk_*.fasta')))} > {fa_out}")
        for p in ESM_DIR.glob("esm_chunk_*.fasta"): p.unlink()
        log(f"Done in {time.time()-t0:.0f}s")

# ── Stage 2: cluster_afdb ─────────────────────────────────────────────────────

def stage_cluster_afdb():
    log("=== cluster_afdb ===")
    afdb_fa    = AFDB_DIR    / "afdb_v6_seqs.fa"
    uniref_xml = UNIPROT_DIR / "uniref50.xml"
    reps_fa    = WORK_DIR    / "AFDBv6_uniref50_reps.fasta"
    replaced   = WORK_DIR    / "AFDBv6_uniref50_rep_replacements.tsv"

    uniparc_tsv = WORK_DIR / "uniparc_afdb_clusters.tsv"

    if not exists(reps_fa, "UniRef50→AFDB rep matching"):
        log("Indexing AFDB accessions...")
        afdb_accs = set()
        with open(afdb_fa) as f:
            for line in f:
                if line.startswith(">"):
                    m = re.search(r"UA=(\S+)", line)
                    if m: afdb_accs.add(m.group(1))
        log(f"  {len(afdb_accs):,} AFDB accessions indexed")

        log("Parsing UniRef50 XML...")
        chosen = {}; replacements = {}
        cluster_id = rep_acc = None
        in_rep = False; n_clusters = 0
        current_uniparc = []

        with open(uniparc_tsv, "w") as upi_out:
            upi_out.write("uniparc_id\tcluster_id\n")
            for event, elem in ET.iterparse(uniref_xml, events=("start", "end")):
                tag = elem.tag.split("}")[-1]
                if event == "start":
                    if tag == "entry":
                        cluster_id = elem.get("id", "").replace("UniRef50_", "")
                        rep_acc = None; in_rep = False; current_uniparc = []
                    elif tag == "representativeMember": in_rep = True
                    elif tag == "member":               in_rep = False
                    elif tag == "dbReference" and elem.get("type") == "UniParc ID":
                        current_uniparc.append(elem.get("id"))
                elif event == "end":
                    if tag == "property" and elem.get("type") == "UniProtKB accession":
                        acc = elem.get("value")
                        if in_rep:
                            rep_acc = acc
                        elif cluster_id not in chosen and acc in afdb_accs:
                            chosen[cluster_id] = acc; replacements[cluster_id] = (rep_acc, acc)
                    elif tag == "representativeMember":
                        in_rep = False
                        if rep_acc in afdb_accs: chosen[cluster_id] = rep_acc
                    elif tag == "entry":
                        n_clusters += 1
                        if cluster_id in chosen:
                            upi_out.writelines(f"{u}\t{cluster_id}\n" for u in current_uniparc)
                        elem.clear()
                        if n_clusters % 5_000_000 == 0:
                            log(f"  {n_clusters:,} clusters, {len(chosen):,} matched...")

        log(f"  {n_clusters:,} clusters; {len(chosen):,} with AFDB coverage; "
            f"{len(replacements):,} reps replaced by members")

        with open(replaced, "w") as f:
            f.write("cluster_id\toriginal_rep\treplacement_member\n")
            f.writelines(f"{c}\t{o}\t{r}\n" for c, (o, r) in sorted(replacements.items()))

        log("Extracting representative sequences...")
        chosen_accs = set(chosen.values()); written = 0; keep = False
        with open(afdb_fa) as fin, open(reps_fa, "w") as fout:
            for line in fin:
                if line.startswith(">"):
                    m = re.search(r"UA=(\S+)", line)
                    keep = bool(m and m.group(1) in chosen_accs)
                if keep:
                    fout.write(line)
                    if line.startswith(">"): written += 1
        log(f"  {written:,} sequences written to {reps_fa}")

# ── Stage 3a: prep_compare ───────────────────────────────────────────────────

def stage_prep_compare():
    log("=== prep_compare ===")
    esm_fa      = ESM_DIR  / "esm_plddt60.fasta"
    afdb_reps   = WORK_DIR / "AFDBv6_uniref50_reps.fasta"
    uniparc_tsv = WORK_DIR / "uniparc_afdb_clusters.tsv"
    afdb_db     = WORK_DIR / "afdb_uniref50_reps_db.dmnd"
    esm_no_upi  = WORK_DIR / "esm_plddt60_no_uniparc.fasta"

    if not exists(esm_no_upi, "UniParc pre-filter"):
        with open(uniparc_tsv) as f:
            next(f)
            afdb_upi = {l.split("\t")[0] for l in f}
        log(f"  {len(afdb_upi):,} UniParc IDs in AFDB clusters")
        n_kept = n_excl = 0
        with open(esm_fa) as fin, open(esm_no_upi, "w") as fout:
            keep = True
            for line in fin:
                if line.startswith(">"):
                    p = line[1:].rstrip().split("|")
                    keep = not (len(p) >= 3 and p[2] == "uniparc" and p[1] in afdb_upi)
                    n_excl += not keep; n_kept += keep
                if keep: fout.write(line)
        log(f"  {n_excl:,} excluded by UniParc; {n_kept:,} remaining")

    if not exists(afdb_db, "diamond makedb"):
        run(f"diamond makedb --in {afdb_reps} --db {WORK_DIR/'afdb_uniref50_reps_db'} --threads 8",
            "diamond makedb...")

# ── Stage 3b: compare_esm_afdb ────────────────────────────────────────────────

def stage_compare_esm_afdb(threads, chunk):
    log(f"=== compare_esm_afdb: {chunk} ===")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    chunk_path  = Path(chunk)
    afdb_db     = WORK_DIR / "afdb_uniref50_reps_db.dmnd"
    chunk_hits  = WORK_DIR / "chunk_hits"
    chunk_hits.mkdir(exist_ok=True)
    hits_tsv    = chunk_hits / f"{chunk_path.stem}.tsv"
    dmnd_log    = chunk_hits / f"{chunk_path.stem}.log"

    if not exists(hits_tsv, "diamond blastp"):
        log(f"  progress → {dmnd_log}")
        with open(dmnd_log, "w") as dlog:
            subprocess.run(
                f"diamond blastp --query {chunk_path} --db {afdb_db} --out {hits_tsv} "
                f"--outfmt 6 qseqid sseqid pident qcovhsp --threads {threads} "
                f"--block-size 5 --evalue 1e-10 --max-target-seqs 1 --tmpdir {TMP_DIR}",
                shell=True, check=True, stderr=dlog)

# ── Stage 4: merge_and_cluster_esmatlas ──────────────────────────────────────

def stage_merge_and_cluster_esmatlas(threads):
    log("=== merge_and_cluster_esmatlas ===")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    esm_no_upi  = WORK_DIR / "esm_plddt60_no_uniparc.fasta"
    hits_merged = WORK_DIR / "esm_plddt60_vs_afdbv6-50.tsv"
    excluded    = WORK_DIR / "esm_excluded.txt"
    non_afdb    = WORK_DIR / "esm_plddt60_non-afdb.fasta"
    clusters = WORK_DIR / "esm_plddt60_non-afdb_clusters.tsv"
    rep_oids = WORK_DIR / "rep_oids.txt"
    reps     = WORK_DIR / "esm_plddt60_non-afdb_reps.fasta"
    reps_zst = WORK_DIR / "esm_plddt60_non-afdb_reps.fasta.zst"

    # merge chunk hits
    if not exists(hits_merged, "merge chunk hits"):
        chunk_tsvs = sorted((WORK_DIR / "chunk_hits").glob("*.tsv"))
        if not chunk_tsvs:
            raise RuntimeError("No chunk hits TSVs found in chunk_hits/")
        run(f"cat {' '.join(str(p) for p in chunk_tsvs)} > {hits_merged}",
            f"Merging {len(chunk_tsvs)} chunk hits...")

    if not exists(excluded, "exclusion list"):
        excl = {p[0] for line in open(hits_merged)
                if len(p := line.split("\t")) >= 4
                and float(p[2]) >= 50 and float(p[3]) >= 80}
        open(excluded, "w").write("\n".join(sorted(excl)) + "\n")
        log(f"  {len(excl):,} excluded (pident>=50, qcovhsp>=80)")

    if not exists(non_afdb, "seqkit grep"):
        run(f"seqkit grep -v -f {excluded} {esm_no_upi} -o {non_afdb}",
            "Removing AFDB-covered sequences...")

    if not exists(rep_oids, "extract rep OIDs"):
        n_reps = 0
        with open(clusters) as fin, open(rep_oids,"w") as fout:
            for line in fin:
                f = line.split("\t")
                if len(f) >= 2 and f[0].strip() == f[1].strip():
                    fout.write(f[0].strip()+"\n"); n_reps += 1
        log(f"  {n_reps:,} representatives")

    if not exists(reps, "extract rep sequences"):
        rep_names = {l.strip() for l in open(rep_oids) if l.strip()}
        written, keep = 0, False
        with open(non_afdb) as fin, open(reps,"w") as fout:
            for line in fin:
                if line.startswith(">"):
                    keep = line[1:].split()[0] in rep_names
                    if keep: written += 1
                if keep:
                    fout.write(line)
        log(f"  {written:,} sequences -> {reps}")

    if not exists(reps_zst, "compress"):
        run(f"zstd -T{threads} {reps} -o {reps_zst}", "Compressing reps...")

# ── Stage 5: linclust_worker ─────────────────────────────────────────────────

def stage_linclust_worker(threads):
    """Run as 40 parallel bsub jobs sharing TMP_DIR; Diamond coordinates internally."""
    log("=== linclust_worker ===")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    non_afdb = WORK_DIR / "esm_plddt60_non-afdb.fasta"
    clusters = WORK_DIR / "esm_plddt60_non-afdb_clusters.tsv"
    if exists(clusters, "linclust"):
        return
    run(f"diamond linclust -d {non_afdb} -o {clusters} -M 32G "
        f"--approx-id {APPROX_ID} --member-cover {MEMBER_COVER} "
        f"--threads {threads} --parallel-tmpdir {TMP_DIR}",
        f"linclust worker (threads={threads})...")

# ── Main ──────────────────────────────────────────────────────────────────────

CONDA_ENV = "at_dbs_setup"

def ensure_env(stage):
    if stage == "setup_env": return
    if CONDA_ENV not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print(f"[re-launching inside conda env '{CONDA_ENV}']", flush=True)
        sys.exit(subprocess.run(
            ["conda", "run", "-n", CONDA_ENV, sys.executable] + sys.argv
        ).returncode)

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["setup_env","download","cluster_afdb","prep_compare",
                            "compare_esm_afdb","linclust_worker","merge_and_cluster_esmatlas"])
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--chunk", default=None, help="Path to chunk fasta (compare_esm_afdb only)")
    args = p.parse_args()
    ensure_env(args.stage)
    if   args.stage == "setup_env":                  stage_setup_env()
    elif args.stage == "download":                   stage_download(args.threads)
    elif args.stage == "cluster_afdb":               stage_cluster_afdb()
    elif args.stage == "prep_compare":              stage_prep_compare()
    elif args.stage == "compare_esm_afdb":
        if not args.chunk:
            p.error("--chunk required for compare_esm_afdb")
        stage_compare_esm_afdb(args.threads, args.chunk)
    elif args.stage == "linclust_worker":             stage_linclust_worker(args.threads)
    elif args.stage == "merge_and_cluster_esmatlas": stage_merge_and_cluster_esmatlas(args.threads)

if __name__ == "__main__":
    main()
