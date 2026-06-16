#!/usr/bin/env python3
"""
AT_classA_kmer.py  —  AlphaTracer test version: kmer sketch search replaces DIAMOND

Class A definition (same as AT_classA.py):
  - No gaps in reference within aligned region
  - No internal gaps in query (flanking gaps allowed)
  - Every 40-residue window >= 80% sequence similarity

Pipeline:
  1. Filter input FASTA  (remove sequences with 'X', or > 2000 aa)
  2. Kmer sketch search  (Rust search binary against afdb_v6_reps sketch index)
  3. Individual NW align (exact per-pair alignment; classify as Class A)
  4. Download ref PDBs   (AlphaFold structures for Class A hits)
  5. Build output PDBs   (copy reference coordinates; replace residue names)

Usage:
  python AT_classA_kmer.py -i proteins.fasta [-t 8] [--top-k 5]
"""

import gzip, os, re, subprocess, sys, time, argparse, pickle, threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import polars as pl
import parasail
import pycurl
import gemmi
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def _tqdm(it, **kw):
    return tqdm(it, **kw) if HAS_TQDM else it


# ── Paths ──────────────────────────────────────────────────────────────────────

AFDB_DIR   = os.environ.get('AT_AFDB_DIR', os.getcwd())
REPS_PQ    = os.path.join(AFDB_DIR, "afdb_v6_reps.pq")
SIDX_CACHE = os.path.join(AFDB_DIR, "afdb_v6_reps_sketches.sidx")
ANN_CACHE  = os.path.join(AFDB_DIR, "afdb_v6_reps_ann_cache.pkl")  # progressive cross-run cache

# Set by _configure_db() in main() — always called, for both default and --sketch-db paths.
_ID_COL      = None   # name of the ID column in REPS_PQ
_HAS_ANN     = True   # False for DBs without function/family/group_size/n_reps columns
_HAS_DB_TYPE = False  # True for merged AFDB+ESMAtlas DBs with db_type column; auto-detected

# Directory containing esm_query.py — override with AT_ESM_DIR env var.
_ESM_DIR = os.environ.get('AT_ESM_DIR',
               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', '..', 'Data', 'ESMAtlas'))


def _configure_db(sketch_db: str | None = None) -> None:
    """Probe parquet schema and set _ID_COL / _HAS_ANN / _HAS_DB_TYPE.

    If sketch_db is given, also redirects REPS_PQ / SIDX_CACHE / ANN_CACHE to
    that path.  Called unconditionally in main() so merged AFDB+ESMAtlas parquets
    are auto-detected even on the default path.
    """
    global REPS_PQ, SIDX_CACHE, ANN_CACHE, _ID_COL, _HAS_ANN, _HAS_DB_TYPE
    if sketch_db is not None:
        path = os.path.expanduser(sketch_db)
        if not os.path.isfile(path):
            sys.exit(f"[FATAL] --sketch-db: file not found: {path}")
        REPS_PQ    = path
        base       = path.rsplit('.', 1)[0]   # strip .pq / .parquet
        SIDX_CACHE = base + ".sidx"
        ANN_CACHE  = base + "_ann_cache.pkl"
    if not os.path.isfile(REPS_PQ):
        sys.exit(f"[FATAL] sequence database not found: {REPS_PQ}\n"
                 f"        Set AT_AFDB_DIR or pass --sketch-db.")
    schema = pq.read_schema(REPS_PQ)
    names  = schema.names
    if "rep_AFDB_ID" in names:
        _ID_COL = "rep_AFDB_ID"
    elif "AFDB_ID" in names:
        _ID_COL = "AFDB_ID"
    elif "header" in names:
        _ID_COL = "header"
    else:
        sys.exit(f"[FATAL] parquet has no recognised ID column "
                 f"(expected rep_AFDB_ID, AFDB_ID, or header). Found: {names}")
    _HAS_ANN     = all(c in names for c in ("function", "family", "group_size", "n_reps"))
    _HAS_DB_TYPE = "db_type" in names

# sketch-rs source is bundled in the package; binaries are compiled into a user cache dir.
_SKETCH_RS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sketch_rs')
_SKETCH_RS_CARGO_TARGET = os.path.join(
    os.environ.get('AT_CACHE_DIR', os.path.expanduser("~/.cache/alphatracer")),
    'sketch_rs',
)
_SKETCH_RS_BIN = os.path.join(_SKETCH_RS_CARGO_TARGET, 'release')

K          = 9
MAX_FREQ   = 0.001
MIN_SHARED = 2

AFDB_VERSION = 6


# ── Constants (same as AT_classA.py) ──────────────────────────────────────────

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='AlphaTracer (kmer test) — Class A pipeline with sketch search'
    )
    p.add_argument('-i', '--input', required=True,
                   help='Input FASTA of query protein sequences')
    p.add_argument('-t', '--threads', type=int, default=4,
                   help='CPU threads for parallel downloads (default: 4)')
    p.add_argument('--top-k', type=int, default=5,
                   help='Hits per query from kmer search (default: 5)')
    p.add_argument('--window-size', type=int, default=40,
                   help='Sliding window size for identity check (default: 40)')
    p.add_argument('--pctsim', type=float, default=80.0,
                   help='Minimum %% sequence similarity per window (default: 80)')
    p.add_argument('--outdir', default=None,
                   help='Output directory (default: AT_processing_<stem>_kmer)')
    p.add_argument('--classify-only', action='store_true',
                   help='Run steps 1-3 only (filter/kmer/classify), write classA.pq, then exit')
    p.add_argument('--download-build-only', action='store_true',
                   help='Skip steps 1-3, read existing classA.pq, run download+build only')
    p.add_argument('--doctest', action='store_true',
                   help='Run doctests and exit')
    p.add_argument('--sketch-db', default=None, metavar='PATH',
                   help='Path to a custom sketch database parquet (e.g. ESMAtlas). '
                        'Must have columns [<id>, sequence] where <id> is one of '
                        'rep_AFDB_ID / AFDB_ID / header. Overrides AT_AFDB_DIR defaults.')
    return p.parse_args()


# ── Alignment helpers (identical to AT_classA.py) ─────────────────────────────

def align_nw(qseqid, sseqid, qseq, sseq):
    """Align qseq vs sseq with Needleman-Wunsch (BLOSUM45, gap open 10, extend 1).

    >>> r = align_nw('q1', 's1', 'MKLVF', 'MKLVF')
    >>> r[2], r[3], r[4]
    ('MKLVF', 'MKLVF', '|||||')
    """
    try:
        r = parasail.nw_trace_striped_16(qseq, sseq, 10, 1, parasail.blosum45)
        comp = r.traceback.comp.replace(' ', '-')
        return qseqid, sseqid, r.traceback.query, r.traceback.ref, comp
    except Exception as e:
        print(f'  Alignment failed for {qseqid}: {e}')
        return None


def fix_hanging_group_letters(seq):
    """Shift 1–3 edge residues into adjacent flanking gaps.

    >>> fix_hanging_group_letters('A----TGC')
    '----ATGC'
    >>> fix_hanging_group_letters('ATGGTT---C')
    'ATGGTTC---'
    """
    left = re.match(r'^([^-]{1,3})(-+)', seq)
    if left and len(left.group(2)) >= 3:
        letters, dashes = left.groups()
        seq = '-' * len(dashes) + letters + seq[len(letters) + len(dashes):]
    right = re.search(r'(-+)([^-]{1,3})$', seq)
    if right and len(right.group(1)) >= 3:
        dashes, letters = right.groups()
        seq = seq[:-(len(dashes) + len(letters))] + letters + '-' * len(dashes)
    return seq


def all_windows_pass(comp_core, window, threshold):
    """Return True if every sliding window has >= threshold similarity.

    >>> all_windows_pass('||||||||||||||||||||', 10, 0.8)
    True
    >>> all_windows_pass('|||||-----||||||||||', 10, 0.8)
    False
    """
    n = len(comp_core)
    if n == 0:
        return False
    if n < window:
        return sum(c in '|:' for c in comp_core) / n >= threshold
    for i in range(n - window + 1):
        w = comp_core[i:i + window]
        if sum(c in '|:' for c in w) / window < threshold:
            return False
    return True


def is_classA(qseq_alg, sseq_alg, alg_comp, window, threshold):
    """Return True if aligned pair qualifies as Class A.

    >>> is_classA('MKLVF',  'MKLVF',  '|||||',    40, 0.8)
    True
    >>> is_classA('MK-LVF', 'MKNLVF', '||~|||',   40, 0.8)
    False
    """
    if '-' in sseq_alg:
        return False
    qseq_core = qseq_alg.strip('-')
    if not qseq_core or '-' in qseq_core:
        return False
    n_leading  = len(qseq_alg) - len(qseq_alg.lstrip('-'))
    n_trailing = len(qseq_alg) - len(qseq_alg.rstrip('-'))
    comp_core  = (alg_comp[n_leading:-n_trailing]
                  if n_trailing > 0 else alg_comp[n_leading:])
    return all_windows_pass(comp_core, window, threshold)


# ── AFDB helpers (identical to AT_classA.py) ──────────────────────────────────

def get_afdb_id(sseqid):
    """Extract bare AlphaFold accession from sseqid.

    >>> get_afdb_id('AF-A0A000-F1')
    'AF-A0A000-F1'
    >>> get_afdb_id('AF-A0A000-F1-model_v4')
    'AF-A0A000-F1'
    >>> get_afdb_id('unknown') is None
    True
    """
    if ':' in sseqid:
        acc = sseqid.split(':')[1]
    elif sseqid.startswith('AF-'):
        acc = sseqid
    else:
        return None
    return re.sub(r'-model_v\d+$', '', acc)


def afdb_local_pdb(afdb_id, pdb_dir):
    return os.path.join(pdb_dir, f'{afdb_id}-model_v{AFDB_VERSION}.pdb')


def _is_valid_pdb(path):
    try:
        with open(path, 'rb') as f:
            header = f.read(6).decode('ascii', errors='ignore')
        return header.strip()[:6] in ('HEADER', 'REMARK', 'ATOM  ', 'MODEL ')
    except Exception:
        return False


def _fetch_pdb(afdb_id, pdb_dir):
    path = afdb_local_pdb(afdb_id, pdb_dir)
    filename = os.path.basename(path)
    if os.path.exists(path) and _is_valid_pdb(path):
        return f'exists:{filename}'
    url = f'https://alphafold.ebi.ac.uk/files/{filename}'
    try:
        with open(path, 'wb') as f:
            c = pycurl.Curl()
            c.setopt(c.URL, url)
            c.setopt(c.WRITEDATA, f)
            c.perform()
            c.close()
        if _is_valid_pdb(path):
            return f'downloaded:{filename}'
        os.remove(path)
        return f'failed:{afdb_id}:server returned non-PDB content'
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return f'failed:{afdb_id}:{e}'


# ── Stage 1: filter FASTA ──────────────────────────────────────────────────────

def stage_filter(input_path, output_path):
    """Write filtered FASTA (no 'X', <= 2000 aa). Returns (n_in, n_out, seq_dict)."""
    filtered = []
    seq_dict = {}
    n_in = 0
    print("  Parsing input FASTA...", flush=True)
    open_fn = gzip.open if input_path.endswith('.gz') else open
    with open_fn(input_path, 'rt') as _fh:
        records = list(SeqIO.parse(_fh, 'fasta'))
    for record in records:
        n_in += 1
        seq = str(record.seq)
        if 'X' not in seq and len(seq) <= 2000:
            filtered.append(f'>{record.id}\n{seq}')
            seq_dict[record.id] = seq
    with open(output_path, 'w') as f:
        f.write('\n'.join(filtered))
    return n_in, len(filtered), seq_dict


# ── Annotation cache ──────────────────────────────────────────────────────────
# Progressive on-disk pickle: accumulates {rep_AFDB_ID: (function, family, group_size, n_reps, sequence)}
# across runs. DuckDB is only queried for IDs not already cached.

def _load_ann_cache() -> dict:
    if os.path.exists(ANN_CACHE):
        try:
            with open(ANN_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}

def _save_ann_cache(cache: dict) -> None:
    tmp = ANN_CACHE + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ANN_CACHE)


# ── Stage 2: kmer sketch search ────────────────────────────────────────────────

def _ensure_binaries():
    index_seqs = os.path.join(_SKETCH_RS_BIN, "index-seqs")
    search     = os.path.join(_SKETCH_RS_BIN, "search")
    if os.path.exists(index_seqs) and os.path.exists(search):
        return index_seqs, search
    print("  sketch-rs binaries not found — building with cargo (one-time, ~30s)...", flush=True)
    os.makedirs(_SKETCH_RS_BIN, exist_ok=True)
    # CARGO_TARGET_DIR redirects build output to the user cache dir,
    # keeping the bundled source (which may be in read-only site-packages) untouched.
    env = {**os.environ, 'CARGO_TARGET_DIR': _SKETCH_RS_CARGO_TARGET}
    r = subprocess.run(["cargo", "build", "--release"], cwd=_SKETCH_RS_SRC,
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit('ERROR: cargo build failed. Is the Rust toolchain installed? '
                 'Install from https://rustup.rs')
    print("  sketch-rs binaries built and cached.", flush=True)
    return index_seqs, search


def _build_sidx(index_seqs_bin):
    """Build V2 .sidx inverted index directly from reps.pq (one step, no sketch parquet).

    The index-seqs binary requires the ID column to be named 'AFDB_ID' or 'rep_AFDB_ID'.
    If the source parquet uses a different name (e.g. 'header' for ESMAtlas), a renamed
    temp parquet is streamed first, then cleaned up after indexing.
    """
    global _ID_COL
    n = pq.read_metadata(REPS_PQ).num_rows
    print(f"  Building V2 inverted index from {n:,} sequences (runs once, cached)...", flush=True)
    t0 = time.time()

    id_col = _ID_COL or "rep_AFDB_ID"
    if id_col in ("rep_AFDB_ID", "AFDB_ID"):
        src_pq = REPS_PQ
        tmp_pq = None
    else:
        # Stream a renamed copy so index-seqs sees the expected column name.
        tmp_pq = REPS_PQ.rsplit('.', 1)[0] + "_idrenamed_tmp.pq"
        print(f"  Streaming renamed parquet ('{id_col}' → 'AFDB_ID') → {tmp_pq}", flush=True)
        reader = pq.ParquetFile(REPS_PQ)
        writer = None
        done   = 0
        for batch in reader.iter_batches(batch_size=100_000):
            cols  = batch.schema.names
            renamed = batch.rename_columns(
                ["AFDB_ID" if c == id_col else c for c in cols]
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp_pq, renamed.schema, compression="zstd")
            writer.write_batch(renamed)
            done += len(batch)
            print(f"    {done:>12}/{n}\r", end="", flush=True)
        if writer:
            writer.close()
        print(flush=True)
        src_pq = tmp_pq

    try:
        r = subprocess.run([index_seqs_bin, src_pq, SIDX_CACHE,
                            str(K), str(MAX_FREQ), "100"])
        if r.returncode != 0:
            print("index-seqs binary failed"); sys.exit(1)
    finally:
        if tmp_pq and os.path.exists(tmp_pq):
            os.remove(tmp_pq)

    print(f"  Done  [{time.time()-t0:.1f}s]", flush=True)


def stage_kmer_search(filtered_fasta, query_seq_dict, outfile, top_k):
    """
    Run Rust kmer sketch search against the AFDB rep index.

    Returns a polars DataFrame with columns matching what stage_align_and_classify expects:
      approx_pident, sseqid, qseqid, evalue, slen, qlen,
      full_qseq, full_sseq, function, family, group_size, n_reps
    """
    index_seqs_bin, search_bin = _ensure_binaries()

    if not os.path.exists(SIDX_CACHE):
        _build_sidx(index_seqs_bin)
    else:
        print(f"  Using cached inverted index: {SIDX_CACHE}", flush=True)

    print(f"  Running Rust search (top_k={top_k})...", flush=True)
    t0 = time.time()
    r = subprocess.run(
        [search_bin, SIDX_CACHE, filtered_fasta,
         str(top_k), str(MIN_SHARED), "150", str(K)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    sys.stderr.write(r.stderr)
    lines = r.stdout.splitlines()
    raw_hits = lines[1:] if lines else []
    print(f"  {len(raw_hits)} hits in {time.time()-t0:.1f}s", flush=True)

    _extra_schema = {
        'db_type': pl.String, 'afdb_id': pl.String,
        'protein_hash': pl.String, 'fragment_id': pl.Int32, 'frag_row': pl.Int32,
    } if _HAS_DB_TYPE else {}

    if not raw_hits:
        return pl.DataFrame(schema={
            'approx_pident': pl.Float64, 'sseqid': pl.String, 'qseqid': pl.String,
            'evalue': pl.Float64, 'slen': pl.Int32, 'qlen': pl.Int32,
            'full_qseq': pl.String, 'full_sseq': pl.String,
            'function': pl.String, 'family': pl.String,
            'group_size': pl.Int32, 'n_reps': pl.Int32,
            **_extra_schema,
        })

    parsed = []
    targets = set()
    for line in raw_hits:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        query, target, shared, jaccard = parts[0], parts[1], int(parts[2]), float(parts[3])
        parsed.append((query, target, shared, jaccard))
        targets.add(target)

    # ── Progressive annotation cache ──────────────────────────────────────────
    # Load what we already know; query DuckDB only for genuinely new targets.
    id_col = _ID_COL or "rep_AFDB_ID"
    ann_pkl = _load_ann_cache()
    missing = targets - ann_pkl.keys()
    print(f"  Annotations: {len(targets)} targets "
          f"({len(targets) - len(missing)} from cache, {len(missing)} new)...", flush=True)
    if missing:
        t_ann = time.time()
        id_list = "', '".join(missing)
        con = duckdb.connect()
        if _HAS_DB_TYPE:
            # Merged DB: fetch db_type + ESM coords + annotations + sequence
            db_rows = con.execute(f"""
                SELECT {id_col}, db_type, afdb_id, protein_hash, fragment_id, frag_row,
                       function, family, group_size, n_reps, sequence
                FROM read_parquet('{REPS_PQ}')
                WHERE {id_col} IN ('{id_list}')
            """).fetchall()
            for row in db_rows:
                # (db_type, afdb_id, protein_hash, frag_id, frag_row, func, fam, gs, nr, seq)
                ann_pkl[row[0]] = row[1:]
        elif _HAS_ANN:
            db_rows = con.execute(f"""
                SELECT {id_col}, function, family, group_size, n_reps, sequence
                FROM read_parquet('{REPS_PQ}')
                WHERE {id_col} IN ('{id_list}')
            """).fetchall()
            for row in db_rows:
                ann_pkl[row[0]] = row[1:]   # (function, family, group_size, n_reps, sequence)
        else:
            db_rows = con.execute(f"""
                SELECT {id_col}, sequence
                FROM read_parquet('{REPS_PQ}')
                WHERE {id_col} IN ('{id_list}')
            """).fetchall()
            for row in db_rows:
                ann_pkl[row[0]] = ("", "", 0, 0, row[1])  # (function, family, group_size, n_reps, sequence)
        con.close()
        _save_ann_cache(ann_pkl)
        print(f"  DuckDB query: {len(missing)} new targets in {time.time()-t_ann:.1f}s  "
              f"(cache now {len(ann_pkl):,} entries)", flush=True)

    if _HAS_DB_TYPE:
        # ann_pkl value: (db_type, afdb_id, protein_hash, frag_id, frag_row, func, fam, gs, nr, seq)
        ann_map  = {t: ann_pkl[t][5:9] for t in targets if t in ann_pkl}   # func,fam,gs,nr
        seq_map  = {t: ann_pkl[t][9]   for t in targets if t in ann_pkl}
        meta_map = {t: ann_pkl[t][:5]  for t in targets if t in ann_pkl}   # db_type..frag_row
    else:
        ann_map  = {t: ann_pkl[t][:4] for t in targets if t in ann_pkl}
        seq_map  = {t: ann_pkl[t][4]  for t in targets if t in ann_pkl}
        meta_map = {}

    # Save raw search output for reference
    with open(outfile, 'w') as f:
        f.write("query\ttarget\tshared\tjaccard\tfunction\tfamily\tsequence\n")
        for query, target, shared, jaccard in parsed:
            func, fam, gs, nr = ann_map.get(target, ("", "", 0, 0))
            seq = seq_map.get(target, "")
            f.write(f"{query}\t{target}\t{shared}\t{jaccard:.4f}\t{func}\t{fam}\t{seq}\n")

    rows = []
    for query, target, shared, jaccard in parsed:
        func, fam, gs, nr = ann_map.get(target, ("", "", 0, 0))
        full_qseq = query_seq_dict.get(query, "")
        full_sseq = seq_map.get(target, "")
        if not full_qseq or not full_sseq:
            continue
        row = {
            'approx_pident': shared,
            'sseqid':        target,
            'qseqid':        query,
            'evalue':        0.0,
            'slen':          len(full_sseq),
            'qlen':          len(full_qseq),
            'full_qseq':     full_qseq,
            'full_sseq':     full_sseq,
            'function':      func,
            'family':        fam,
            'group_size':    gs,
            'n_reps':        nr,
        }
        if _HAS_DB_TYPE:
            db_type, afdb_id, protein_hash, frag_id, frag_row = meta_map.get(
                target, ('afdb', '', '', -1, -1))
            row.update({
                'db_type':      db_type or 'afdb',
                'afdb_id':      afdb_id  or '',
                'protein_hash': protein_hash or '',
                'fragment_id':  int(frag_id)  if frag_id  is not None else -1,
                'frag_row':     int(frag_row) if frag_row is not None else -1,
            })
        rows.append(row)

    return pl.DataFrame(rows)


# ── Stage 3: align and classify ────────────────────────────────────────────────

def stage_align_and_classify(hits_df, window, threshold, threads=4):
    total = len(hits_df)
    print(f"  Aligning {total} query-hit pairs (threads={threads})...", flush=True)
    rows_list = list(hits_df.iter_rows(named=True))

    def _aln(row):
        return align_nw(row['qseqid'], row['sseqid'], row['full_qseq'], row['full_sseq'])

    aln_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for result in ex.map(_aln, rows_list):
            done += 1
            if result:
                aln_rows.append(result)
            if done % 200 == 0 or done == total:
                print(f'  Aligning {done}/{total}...', end='\r', flush=True)
    print()

    if not aln_rows:
        return pl.DataFrame(schema={
            'qseqid': pl.String, 'sseqid': pl.String,
            'qseq_alg': pl.String, 'sseq_alg': pl.String, 'alg_comp': pl.String,
        })

    aln_df = pl.DataFrame(
        aln_rows,
        schema=['qseqid', 'sseqid', 'qseq_alg', 'sseq_alg', 'alg_comp'],
        orient='row',
    )
    for col in ['qseq_alg', 'sseq_alg', 'alg_comp']:
        aln_df = aln_df.with_columns(
            pl.col(col).map_elements(fix_hanging_group_letters, return_dtype=pl.String)
        )

    merged = hits_df.join(aln_df, on=['qseqid', 'sseqid'], how='inner')

    classA_df = merged.filter(
        pl.struct(['qseq_alg', 'sseq_alg', 'alg_comp']).map_elements(
            lambda r: is_classA(r['qseq_alg'], r['sseq_alg'], r['alg_comp'],
                                window, threshold),
            return_dtype=pl.Boolean,
        )
    )
    return classA_df


# ── ESM Atlas structure fetcher ────────────────────────────────────────────────

def _esm_local_pdb(protein_hash, pdb_dir):
    return os.path.join(pdb_dir, f'esm_{protein_hash}.pdb')


def _fetch_esm_pdbs(esm_df, pdb_dir, n_workers=8):
    """Batch-fetch ESM Atlas structure blobs from Lance and write as PDB files."""
    esm_dir = os.path.abspath(_ESM_DIR)
    if esm_dir not in sys.path:
        sys.path.insert(0, esm_dir)
    try:
        import esm_query as _esm
    except ImportError:
        print(f'  [WARN] Cannot import esm_query from {esm_dir} — ESM Atlas hits will be skipped')
        return

    hits = []
    for row in esm_df.unique(subset=['protein_hash']).iter_rows(named=True):
        ph  = row['protein_hash']
        fid = row['fragment_id']
        fr  = row['frag_row']
        if not ph or fid < 0 or fr < 0:
            continue
        if not os.path.exists(_esm_local_pdb(ph, pdb_dir)):
            hits.append({'fragment_id': fid, 'frag_row': fr, 'protein_hash': ph})

    if not hits:
        return

    print(f'  Fetching {len(hits)} ESM Atlas structure(s) from S3...', flush=True)
    t0 = time.time()
    try:
        results = _esm.query_from_hits(hits, columns=['protein_hash', 'structure_blob'],
                                        n_workers=n_workers)
    except Exception as e:
        print(f'  [WARN] ESM Atlas fetch failed: {type(e).__name__}: {e}')
        return

    n_written = 0
    for batch in results.to_batches():
        hashes = batch['protein_hash'].to_pylist()
        blobs  = batch['structure_blob'].to_pylist()
        for ph, blob in zip(hashes, blobs):
            pdb_path = _esm_local_pdb(ph, pdb_dir)
            try:
                with open(pdb_path, 'w') as f:
                    f.write(_esm.blob_to_pdb(blob))
                n_written += 1
            except Exception as e:
                print(f'  [WARN] ESM Atlas decode failed for {ph}: {e}')

    print(f'  Fetched {n_written}/{len(hits)} ESM Atlas structure(s) in {time.time()-t0:.1f}s',
          flush=True)


# ── Stage 4: download PDBs ─────────────────────────────────────────────────────

def stage_download(classA_df, pdb_dir, threads):
    top_sseqids = (
        classA_df
        .sort('approx_pident', descending=True)
        .group_by('qseqid', maintain_order=True)
        .agg(pl.col('sseqid').first())
        ['sseqid'].to_list()
    )
    afdb_ids = {get_afdb_id(sid) for sid in top_sseqids if get_afdb_id(sid)}
    print(f'  {len(afdb_ids)} unique AlphaFold accession(s) to download (v{AFDB_VERSION})',
          flush=True)

    afdb_list = list(afdb_ids)

    # ── first pass (threaded) ──────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(_fetch_pdb, aid, pdb_dir): aid for aid in afdb_list}
        results = {}
        pbar = _tqdm(as_completed(futures), total=len(afdb_list), unit="pdb") \
               if HAS_TQDM else as_completed(futures)
        for fut in pbar:
            aid = futures[fut]
            r = fut.result()
            results[aid] = r
            if r.startswith('fail'):
                time.sleep(0.5)

    # ── retry failures ─────────────────────────────────────────────────────
    retry = [aid for aid, r in results.items() if r.startswith('fail')]
    if retry:
        print(f'  Retrying {len(retry)} failed download(s)...')
        for aid in retry:
            r = _fetch_pdb(aid, pdb_dir)
            results[aid] = r
            if r.startswith('fail'):
                time.sleep(0.5)

    all_results = list(results.values())
    summary = Counter(r.split(':')[0] for r in all_results)
    for r in all_results:
        if r.startswith('fail'):
            print(f'  FAILED: {r}')
    print(f"  Downloaded: {summary['downloaded']}  "
          f"Already present: {summary['exists']}  "
          f"Failed: {summary.get('failed', 0) + summary.get('fail', 0)}")


# ── Stage 4+5: pipelined download and build ────────────────────────────────────

def stage_download_and_build(classA_df, pdb_dir, output_pdbs_dir, threads):
    """Download reference PDBs and build output PDBs in a pipelined fashion.

    Each query's output PDB is built immediately after its primary reference PDB
    download completes, so builds overlap with remaining downloads.
    ESM Atlas structures are fetched from Lance S3 before the AFDB download loop.
    """
    # ── Pre-fetch ESM Atlas structures in background (concurrent with AFDB downloads)
    _esm_fetch_thread = None
    if _HAS_DB_TYPE and 'db_type' in classA_df.columns:
        esm_df = classA_df.filter(pl.col('db_type') == 'esm_atlas')
        if len(esm_df) > 0:
            n_esm = esm_df['protein_hash'].n_unique()
            print(f'  Pre-fetching {n_esm} unique ESM Atlas structure(s) (background)...', flush=True)
            _esm_fetch_thread = threading.Thread(
                target=_fetch_esm_pdbs, args=(esm_df, pdb_dir),
                kwargs={'n_workers': 16}, daemon=True)
            _esm_fetch_thread.start()

    query_hits = defaultdict(list)
    for row in classA_df.sort('approx_pident', descending=True).iter_rows(named=True):
        query_hits[row['qseqid']].append(row)

    # Map each afdb_id to the queries for which it is the primary (rank-0) hit.
    # Only AFDB hits have valid afdb_ids; ESM Atlas hits return None from get_afdb_id().
    aid_primary_queries = defaultdict(list)
    for qseqid, hits in query_hits.items():
        primary_aid = get_afdb_id(hits[0]['sseqid'])
        if primary_aid:
            aid_primary_queries[primary_aid].append(qseqid)

    # Only pre-download primary (top-pident) AFDB hits; fallbacks fetched on demand.
    all_aids = {aid for aid in (get_afdb_id(hits[0]['sseqid']) for hits in query_hits.values()) if aid}

    print(f'  {len(all_aids)} unique AlphaFold accession(s) to download (v{AFDB_VERSION})',
          flush=True)

    built = set()
    n_ok = n_fail = 0
    ok_by_db: Counter = Counter()
    fail_by_db: Counter = Counter()
    timings = []
    _counts_lock = __import__('threading').Lock()
    _counts_path = os.path.join(os.path.dirname(output_pdbs_dir), '.classA_db_counts')

    def _write_counts():
        import json, tempfile
        tmp = _counts_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'ok': dict(ok_by_db), 'fail': dict(fail_by_db)}, f)
        os.replace(tmp, _counts_path)

    def _try_build_query(qseqid):
        out_pdb = os.path.join(output_pdbs_dir, f'classA:{qseqid}.pdb')
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            return True, (query_hits[qseqid][0].get('db_type') or 'afdb')
        hits = query_hits[qseqid]
        last_err = 'no hits'
        for rank, row in enumerate(hits):
            db_type = row.get('db_type', 'afdb') or 'afdb'
            if db_type != 'esm_atlas':
                aid = get_afdb_id(row['sseqid'])
                if aid and not os.path.exists(afdb_local_pdb(aid, pdb_dir)):
                    _fetch_pdb(aid, pdb_dir)   # on-demand fetch for fallback AFDB hits
            t0 = time.perf_counter()
            err = _try_build_pdb(row, pdb_dir, out_pdb)
            elapsed = time.perf_counter() - t0
            if err is None:
                if rank > 0:
                    print(f'  {qseqid}: built from fallback hit {rank+1} ({row["sseqid"]})')
                timings.append(elapsed)
                return True, db_type
            last_err = err
            if rank == 0 and len(hits) > 1:
                print(f'  {qseqid}: hit 1 failed ({err}), trying next...')
        print(f'  {qseqid}: all {len(hits)} hit(s) failed — {last_err}')
        return False, (hits[0].get('db_type') or 'afdb')

    results = {}

    # ── first pass: download all, build inline as primary PDBs arrive ─────
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(_fetch_pdb, aid, pdb_dir): aid for aid in all_aids}
        for fut in as_completed(futures):
            aid = futures[fut]
            r = fut.result()
            results[aid] = r
            if r.startswith('fail'):
                time.sleep(0.5)
            else:
                for qseqid in aid_primary_queries.get(aid, []):
                    if qseqid not in built:
                        built.add(qseqid)
                        ok, db = _try_build_query(qseqid)
                        with _counts_lock:
                            if ok:
                                n_ok += 1; ok_by_db[db] += 1
                            else:
                                n_fail += 1; fail_by_db[db] += 1
                            _write_counts()

    # ── wait for ESM background fetch before building ESM queries ─────────────
    if _esm_fetch_thread is not None:
        _esm_fetch_thread.join()

    # ── retry failed downloads ─────────────────────────────────────────────
    retry = [aid for aid, r in results.items() if r.startswith('fail')]
    if retry:
        print(f'  Retrying {len(retry)} failed download(s)...')
        for aid in retry:
            r = _fetch_pdb(aid, pdb_dir)
            results[aid] = r
            if r.startswith('fail'):
                time.sleep(0.5)

    # ── final pass: build any queries whose primary download failed ────────
    for qseqid in query_hits:
        if qseqid not in built:
            built.add(qseqid)
            ok, db = _try_build_query(qseqid)
            with _counts_lock:
                if ok:
                    n_ok += 1; ok_by_db[db] += 1
                else:
                    n_fail += 1; fail_by_db[db] += 1
                _write_counts()

    all_results = list(results.values())
    summary = Counter(r.split(':')[0] for r in all_results)
    for r in all_results:
        if r.startswith('fail'):
            print(f'  FAILED: {r}')
    print(f"  Downloaded: {summary['downloaded']}  "
          f"Already present: {summary['exists']}  "
          f"Failed: {summary.get('failed', 0) + summary.get('fail', 0)}")

    return n_ok, n_fail, timings, ok_by_db, fail_by_db


# ── Stage 5: build output PDBs ─────────────────────────────────────────────────

def _try_build_pdb(row, pdb_dir, out_pdb):
    qseq     = row['full_qseq']
    qseq_alg = row['qseq_alg']
    sseqid   = row['sseqid']

    # Dispatch on db_type when using the merged database
    db_type = row.get('db_type', 'afdb') or 'afdb'
    if db_type == 'esm_atlas':
        protein_hash = row.get('protein_hash') or sseqid.split('|')[0]
        src_pdb = _esm_local_pdb(protein_hash, pdb_dir)
        if not os.path.exists(src_pdb):
            return f'ESM Atlas PDB not cached ({protein_hash})'
    else:
        afdb_id  = row.get('afdb_id') or get_afdb_id(sseqid)
        if not afdb_id:
            return f'cannot parse AFDB id from "{sseqid}"'
        src_pdb = afdb_local_pdb(afdb_id, pdb_dir)
        if not os.path.exists(src_pdb):
            return f'source PDB not found ({afdb_id})'
    n_leading = len(qseq_alg) - len(qseq_alg.lstrip('-'))
    start = n_leading
    end   = start + len(qseq)
    try:
        st    = gemmi.read_structure(src_pdb)
        if db_type == 'esm_atlas':
            st.setup_entities()
        chain = st[0]['A']
        residues = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        if end > len(residues):
            return f'segment [{start}:{end}] exceeds reference length {len(residues)}'
        if len(qseq) != end - start:
            return f'length mismatch — segment {end-start} residues vs query {len(qseq)}'
        for i, res in enumerate(residues[start:end]):
            new_resname = ONE_TO_THREE.get(qseq[i])
            if new_resname:
                res.name = new_resname
        if start > 0 or end < len(residues):
            keep = {id(r) for r in residues[start:end]}
            to_del = [i for i, r in enumerate(chain)
                      if r.entity_type == gemmi.EntityType.Polymer and id(r) not in keep]
            for i in reversed(to_del):
                del chain[i]
        st.write_pdb(out_pdb)
        return None
    except Exception as e:
        return str(e)


def stage_build_pdbs(classA_df, pdb_dir, output_pdbs_dir):
    n_ok = n_fail = 0
    timings = []
    query_hits = defaultdict(list)
    for row in classA_df.sort('approx_pident', descending=True).iter_rows(named=True):
        query_hits[row['qseqid']].append(row)

    items = list(query_hits.items())
    pbar = _tqdm(items, unit="query") if HAS_TQDM else items
    for qseqid, hits in pbar:
        out_pdb = os.path.join(output_pdbs_dir, f'classA:{qseqid}.pdb')
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            n_ok += 1
            continue
        success = False
        last_err = 'no Class A hits available'
        for rank, row in enumerate(hits):
            t0 = time.perf_counter()
            err = _try_build_pdb(row, pdb_dir, out_pdb)
            elapsed = time.perf_counter() - t0
            if err is None:
                if rank > 0:
                    print(f'  {qseqid}: built from fallback hit {rank+1} ({row["sseqid"]})')
                timings.append(elapsed)
                success = True
                break
            else:
                last_err = err
                if rank == 0 and len(hits) > 1:
                    print(f'  {qseqid}: hit 1 failed ({err}), trying next...')
        if success:
            n_ok += 1
        else:
            print(f'  {qseqid}: all {len(hits)} hit(s) failed — {last_err}')
            n_fail += 1

    return n_ok, n_fail, timings


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.doctest:
        import doctest
        results = doctest.testmod(verbose=True)
        raise SystemExit(0 if results.failed == 0 else 1)

    _configure_db(args.sketch_db if args.sketch_db else None)

    input_basename  = Path(args.input).stem
    while '.' in input_basename:
        input_basename = Path(input_basename).stem
    outdir          = args.outdir or f'AT_processing_{input_basename}_kmer'
    pdb_dir         = os.path.join(outdir, 'AF_pdbs')
    output_pdbs_dir = os.path.join(outdir, 'output_pdbs_classA')
    for d in [outdir, pdb_dir, output_pdbs_dir]:
        os.makedirs(d, exist_ok=True)

    print('=' * 60)
    print('AlphaTracer (kmer test)  —  Class A Pipeline')
    print('=' * 60)
    print(f'  Input:       {args.input}')
    print(f'  Search:      kmer sketch (k={K}, n=100 hashes, top_k={args.top_k})')
    print(f'  Index:       {SIDX_CACHE}')
    print(f'  Output:      {outdir}/')
    print(f'  Window:      {args.window_size} aa, >= {args.pctsim:.0f}% similarity')
    print()

    t_total = time.time()

    if args.download_build_only:
        # ── Steps 4-5 only: read existing classA.pq ───────────────────────────
        classA_pq = os.path.join(outdir, 'classA.pq')
        if not os.path.exists(classA_pq):
            sys.exit(f'[FATAL] --download-build-only requires {classA_pq} (run classify phase first)')
        classA_df = pl.read_parquet(classA_pq)
        n_classA_queries = classA_df['qseqid'].n_unique() if len(classA_df) > 0 else 0
        print(f'  Loaded {n_classA_queries} Class A queries from {classA_pq}')

        print('\n[4-5/5] Downloading and building Class A PDBs (pipelined)...')
        t4 = time.time()
        n_ok, n_fail, timings, ok_by_db, fail_by_db = stage_download_and_build(classA_df, pdb_dir, output_pdbs_dir, args.threads)
        print(f'  [{time.time()-t4:.1f}s]')

        print()
        print('=' * 60)
        print('Class A download+build complete.')
        print(f'  PDBs written: {n_ok}  failed: {n_fail}')
        if _HAS_DB_TYPE:
            print(f'    AFDB:       ok={ok_by_db["afdb"]}  failed={fail_by_db["afdb"]}')
            print(f'    ESM Atlas:  ok={ok_by_db["esm_atlas"]}  failed={fail_by_db["esm_atlas"]}')
        print(f'  Total runtime: {time.time()-t_total:.1f}s')
        print('=' * 60)
        return

    # ── 1. Filter ──────────────────────────────────────────────────────────────
    print('[1/5] Filtering input FASTA...')
    filtered_fasta = os.path.join(outdir, 'input_seqs_filtered.fa')
    n_in, n_filtered, query_seq_dict = stage_filter(args.input, filtered_fasta)
    print(f'  {n_in} input → {n_filtered} passed filter')

    # ── 2. Kmer search ────────────────────────────────────────────────────────
    kmer_out = os.path.join(outdir, 'kmer_hits.tsv')
    print(f'\n[2/5] Kmer sketch search...')
    t2 = time.time()
    hits_df = stage_kmer_search(filtered_fasta, query_seq_dict, kmer_out, args.top_k)
    n_queries_hit = hits_df['qseqid'].n_unique() if len(hits_df) > 0 else 0
    hits_df.write_parquet(os.path.join(outdir, 'kmer_hits.pq'))
    hits_df.write_parquet(os.path.join(outdir, 'allhits.pq'))
    print(f'  {len(hits_df)} hits across {n_queries_hit} queries  [{time.time()-t2:.1f}s]')
    print(f'  Raw hits saved to: {kmer_out}')

    if len(hits_df) == 0:
        print('\nNo kmer hits found. Exiting.')
        return

    # Filter out hits with empty sequences (shouldn't happen but guard)
    hits_df = hits_df.filter(
        pl.col('full_qseq').str.len_chars() > 0,
        pl.col('full_sseq').str.len_chars() > 0,
    )

    # ── 3. Align and classify ─────────────────────────────────────────────────
    print(f'\n[3/5] NW alignment and Class A classification '
          f'(window={args.window_size} aa, pctsim={args.pctsim:.0f}%)...')
    t3 = time.time()
    classA_df = stage_align_and_classify(hits_df, args.window_size, args.pctsim / 100, args.threads)
    classA_df.write_parquet(os.path.join(outdir, 'classA.pq'))
    n_classA_queries = classA_df['qseqid'].n_unique() if len(classA_df) > 0 else 0
    print(f'  {n_classA_queries} / {n_queries_hit} queries have >= 1 Class A hit '
          f'({len(classA_df)} total Class A hit-pairs)  [{time.time()-t3:.1f}s]')

    if len(classA_df) == 0:
        print('\nNo Class A sequences found. Exiting.')
        return

    if args.classify_only:
        print('\n[--classify-only] Stopping after classification. classA.pq written.')
        print(f'  Total runtime: {time.time()-t_total:.1f}s')
        return

    # ── 4+5. Download and build PDBs (pipelined) ─────────────────────────────
    print('\n[4-5/5] Downloading and building Class A PDBs (pipelined)...')
    t4 = time.time()
    n_ok, n_fail, timings, ok_by_db, fail_by_db = stage_download_and_build(classA_df, pdb_dir, output_pdbs_dir, args.threads)
    print(f'  [{time.time()-t4:.1f}s]')

    print()
    print('=' * 60)
    print('Class A pipeline complete.')
    print(f'  Class A sequences:  {n_classA_queries}')
    print(f'  PDBs written:       {n_ok}')
    if _HAS_DB_TYPE:
        print(f'    AFDB:           {ok_by_db["afdb"]}')
        print(f'    ESM Atlas:      {ok_by_db["esm_atlas"]}')
    print(f'  PDBs failed:        {n_fail}')
    if _HAS_DB_TYPE and n_fail:
        print(f'    AFDB:           {fail_by_db["afdb"]}')
        print(f'    ESM Atlas:      {fail_by_db["esm_atlas"]}')
    if timings:
        print(f'  Time per structure: min={min(timings):.3f}s  '
              f'mean={sum(timings)/len(timings):.3f}s  '
              f'max={max(timings):.3f}s')
    print(f'  Total runtime:      {time.time()-t_total:.1f}s')
    print(f'  Output directory:   {output_pdbs_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
