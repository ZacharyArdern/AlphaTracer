#!/usr/bin/env python3
"""
AT_classA.py  —  AlphaTracer 1.0, Class A full pipeline

Class A definition
------------------
A query is Class A if its best AFDB hit has NO indels relative to the query:
  - No gaps in the reference (sseq_alg) within the aligned region
  - No internal gaps in the query (qseq_alg) — only leading/trailing dashes
    are allowed, meaning the query may be fully *embedded* in the reference
  - Every sliding window of 40 residues in the aligned core has >= 80%
    sequence similarity ('|' identity + ':' positive matches in the NW comparison)

Pipeline stages (all in this script)
-------------------------------------
  1. Filter input FASTA  (remove sequences with 'X', or > 2000 aa)
  2. DIAMOND blastp      (fast homology search against AFDB)
  3. Individual NW align (exact per-pair alignment; classify as Class A)
  4. Download ref PDBs   (AlphaFold structures for Class A hits)
  5. Build output PDBs   (copy reference coordinates; replace residue names)

Usage
-----
  python AT_classA.py -i proteins.fasta -d AFDB.dmnd [-t 8]
"""

import os
import re
import subprocess
import time
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import polars as pl
import parasail
import pycurl
import gemmi
from Bio import SeqIO


# ── Constants ──────────────────────────────────────────────────────────────────

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='AlphaTracer 1.0 — Class A full pipeline'
    )
    p.add_argument('-i', '--input', required=True,
                   help='Input FASTA of query protein sequences')
    _afdb_dir = os.environ.get('AT_AFDB_DIR', os.path.expanduser('~/Science/Data/AFDB'))
    p.add_argument('-d', '--database',
                   default=os.path.join(_afdb_dir, 'bacarc8080.dmnd'),
                   help='DIAMOND database (default: $AT_AFDB_DIR/bacarc8080.dmnd)')
    p.add_argument('-t', '--threads', type=int, default=4,
                   help='CPU threads for DIAMOND and parallel downloads')
    p.add_argument('--window-size', type=int, default=40,
                   help='Sliding window size for identity check (default: 40)')
    p.add_argument('--pctsim', type=float, default=80.0,
                   help='Minimum %% sequence similarity per 40 aa window (default: 80)')
    p.add_argument('--doctest', action='store_true',
                   help='Run doctests and exit')
    return p.parse_args()


# ── Alignment helpers ──────────────────────────────────────────────────────────

def align_nw(qseqid: str, sseqid: str, qseq: str, sseq: str):
    """Align qseq vs sseq with Needleman-Wunsch (BLOSUM45, gap open 10, extend 1).

    Returns (qseqid, sseqid, qseq_alg, sseq_alg, comp_str), or None on failure.
    comp_str uses '|' for identity, ':' for positive, '-' for gap.

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


def fix_hanging_group_letters(seq: str) -> str:
    """Shift 1–3 edge residues into adjacent flanking gaps to tidy boundaries.

    >>> fix_hanging_group_letters('A----TGC')
    '----ATGC'
    >>> fix_hanging_group_letters('ATGGTT---C')
    'ATGGTTC---'
    >>> fix_hanging_group_letters('A--TGCCG')
    'A--TGCCG'
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


def all_windows_pass(comp_core: str, window: int, threshold: float) -> bool:
    """Return True if every sliding window of `window` residues in comp_core
    has >= threshold fraction of similarity ('|' identity + ':' positive characters).

    Sequences shorter than `window` are checked as a whole.

    >>> all_windows_pass('||||||||||||||||||||', 10, 0.8)
    True
    >>> all_windows_pass('|||||-----||||||||||', 10, 0.8)
    False
    >>> all_windows_pass('||||', 10, 0.8)
    True
    >>> all_windows_pass('', 10, 0.8)
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


def is_classA(qseq_alg: str, sseq_alg: str, alg_comp: str,
              window: int, threshold: float) -> bool:
    """Return True if an aligned pair qualifies as Class A.

    Class A requires:
      1. sseq_alg has no gaps (reference fully present in aligned region)
      2. qseq_alg has no *internal* gaps (only flanking '-' allowed;
         the query may be shorter and embedded within the reference)
      3. Every sliding window of `window` residues in the core alignment
         has >= threshold fraction of identity ('|' chars)

    >>> is_classA('MKLVF',  'MKLVF',  '|||||',    40, 0.8)
    True
    >>> is_classA('--MKLVF', 'ABMKLVF', '--|||||', 40, 0.8)
    True
    >>> is_classA('MK-LVF', 'MKNLVF', '||~|||',   40, 0.8)
    False
    >>> is_classA('MKLVF', 'MK-LVF', '||~|||',    40, 0.8)
    False
    """
    # 1. reference must be gap-free
    if '-' in sseq_alg:
        return False
    # 2. query must have only flanking gaps, no internal gaps
    qseq_core = qseq_alg.strip('-')
    if not qseq_core or '-' in qseq_core:
        return False
    # extract the aligned core of the comparison string
    n_leading = len(qseq_alg) - len(qseq_alg.lstrip('-'))
    n_trailing = len(qseq_alg) - len(qseq_alg.rstrip('-'))
    comp_core = (alg_comp[n_leading:-n_trailing]
                 if n_trailing > 0 else alg_comp[n_leading:])
    # 3. window identity check
    return all_windows_pass(comp_core, window, threshold)


# ── AFDB helpers ───────────────────────────────────────────────────────────────

def get_afdb_id(sseqid: str) -> str | None:
    """Extract the AlphaFold accession from a sseqid field.

    Handles 'sp:AF-XXXX-F1', 'AF-XXXX-F1', and the common DIAMOND DB format
    where the version suffix is already present ('AF-XXXX-F1-model_v4').
    Always returns the bare accession WITHOUT the '-model_v4' version suffix
    so that afdb_url() and afdb_local_pdb() can append it consistently.

    >>> get_afdb_id('sp:AF-A0A000-F1')
    'AF-A0A000-F1'
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
    # Strip any version suffix already present (e.g. -model_v4, -model_v6)
    acc = re.sub(r'-model_v\d+$', '', acc)
    return acc


AFDB_VERSION = 6


def afdb_local_pdb(afdb_id: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'{afdb_id}-model_v{AFDB_VERSION}.pdb')


def afdb_url(afdb_id: str) -> str:
    return f'https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v{AFDB_VERSION}.pdb'


# ── Stage 1: filter FASTA ──────────────────────────────────────────────────────

def stage_filter(input_path: str, output_path: str) -> int:
    """Write filtered FASTA (no 'X', <= 2000 aa). Returns count written."""
    filtered = []
    for record in SeqIO.parse(input_path, 'fasta'):
        if 'X' not in str(record.seq) and len(record.seq) <= 2000:
            filtered.append(f'>{record.id}\n{record.seq}')
    with open(output_path, 'w') as f:
        f.write('\n'.join(filtered))
    return len(filtered)


# ── Stage 2: DIAMOND search ────────────────────────────────────────────────────

DIAMOND_COLS = ['approx_pident', 'sseqid', 'qseqid', 'evalue',
                'slen', 'qlen', 'full_qseq', 'full_sseq']
DIAMOND_NCOLS = len(DIAMOND_COLS)


def diamond_hits_valid(path: str) -> bool:
    """Return True if `path` is a non-empty TSV with the expected 8 columns.

    Reads only the first data line to validate structure without loading the
    whole file.
    """
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return False
    try:
        with open(path) as f:
            first = f.readline()
        return first.count('\t') == DIAMOND_NCOLS - 1
    except Exception:
        return False


def stage_diamond(filtered_fasta: str, database: str, output_file: str,
                  threads: int) -> pl.DataFrame:
    """Run DIAMOND blastp; return all-hits DataFrame."""
    cmd = [
        'diamond', 'blastp', '--fast',
        '-q', filtered_fasta,
        '--db', database,
        '-b', '2',
        '-o', output_file,
        '--threads', str(threads),
        '--comp-based-stats', '2',
        '--masking', '0',
        '-f', '6', 'approx_pident', 'sseqid', 'qseqid', 'evalue',
                   'slen', 'qlen', 'full_qseq', 'full_sseq',
        '--evalue', '1e-5',
        '--max-target-seqs', '3',
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f'DIAMOND failed (exit code {r.returncode})')
    while not diamond_hits_valid(output_file):
        time.sleep(0.2)
    return pl.read_csv(output_file, has_header=False, separator='\t', new_columns=DIAMOND_COLS)


# ── Stage 3: align and classify ────────────────────────────────────────────────

def stage_align_and_classify(tophits_df: pl.DataFrame,
                              window: int, threshold: float) -> pl.DataFrame:
    """Run individual NW alignment for each query/hit pair, then
    filter to Class A using the no-indel + window-identity criteria.

    Returns a DataFrame with alignment columns added.
    """
    aln_rows = []
    total = len(tophits_df)
    for i, row in enumerate(tophits_df.iter_rows(named=True), 1):
        if i % 500 == 0 or i == total:
            print(f'  Aligning {i}/{total}...', end='\r')
        aln = align_nw(row['qseqid'], row['sseqid'],
                       row['full_qseq'], row['full_sseq'])
        if aln:
            aln_rows.append(aln)
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

    # tidy alignment boundaries (shift stray edge residues into flanking gaps)
    for col in ['qseq_alg', 'sseq_alg', 'alg_comp']:
        aln_df = aln_df.with_columns(
            pl.col(col).map_elements(fix_hanging_group_letters, return_dtype=pl.String)
        )

    # join alignment results back onto tophits (keeps full_qseq etc.)
    merged = tophits_df.join(aln_df, on=['qseqid', 'sseqid'], how='inner')

    # classify: keep only Class A rows
    classA_df = merged.filter(
        pl.struct(['qseq_alg', 'sseq_alg', 'alg_comp']).map_elements(
            lambda r: is_classA(
                r['qseq_alg'], r['sseq_alg'], r['alg_comp'],
                window, threshold
            ),
            return_dtype=pl.Boolean,
        )
    )
    return classA_df


# ── Stage 4: download PDBs ─────────────────────────────────────────────────────

def stage_download(classA_df: pl.DataFrame, pdb_dir: str, threads: int):
    """Download AlphaFold PDB files for the top hit per query."""
    top_sseqids = (
        classA_df
        .sort('approx_pident', descending=True)
        .group_by('qseqid', maintain_order=True)
        .agg(pl.col('sseqid').first())
        ['sseqid'].to_list()
    )
    afdb_ids = {
        get_afdb_id(sid)
        for sid in top_sseqids
        if get_afdb_id(sid) is not None
    }
    print(f'  {len(afdb_ids)} unique AlphaFold accession(s) to download (v{AFDB_VERSION})')

    def _is_valid_pdb(path: str) -> bool:
        """Return True if the file looks like a PDB (starts with a PDB record keyword)."""
        try:
            with open(path, 'rb') as f:
                header = f.read(6).decode('ascii', errors='ignore')
            return header.strip()[:6] in ('HEADER', 'REMARK', 'ATOM  ', 'MODEL ')
        except Exception:
            return False

    def _fetch(afdb_id):
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

    with ThreadPoolExecutor(max_workers=min(32, len(afdb_ids))) as ex:
        results = list(ex.map(_fetch, list(afdb_ids)))

    summary = Counter(r.split(':')[0] for r in results)
    for r in results:
        if r.startswith('failed'):
            print(f'  FAILED: {r}')
    print(f"  Downloaded: {summary['downloaded']}  "
          f"Already present: {summary['exists']}  "
          f"Failed: {summary['failed']}")


# ── Stage 5: build output PDBs ─────────────────────────────────────────────────

def _try_build_pdb(row: dict, pdb_dir: str, out_pdb: str) -> str | None:
    """Attempt to build a single output PDB for one hit row.

    Returns None on success, or an error string on failure.
    """
    qseq     = row['full_qseq']
    qseq_alg = row['qseq_alg']
    sseqid   = row['sseqid']

    afdb_id = get_afdb_id(sseqid)
    if afdb_id is None:
        return f'cannot parse AFDB id from "{sseqid}"'

    src_pdb = afdb_local_pdb(afdb_id, pdb_dir)
    if not os.path.exists(src_pdb):
        return f'source PDB not found ({afdb_id})'

    n_leading = len(qseq_alg) - len(qseq_alg.lstrip('-'))
    start     = n_leading
    end       = start + len(qseq)

    try:
        st    = gemmi.read_structure(src_pdb)
        chain = st[0]['A']
        residues = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]

        if end > len(residues):
            return (f'segment [{start}:{end}] exceeds '
                    f'reference length {len(residues)}')

        if len(qseq) != end - start:
            return (f'length mismatch — '
                    f'segment {end - start} residues vs query {len(qseq)}')

        for i, res in enumerate(residues[start:end]):
            new_resname = ONE_TO_THREE.get(qseq[i])
            if new_resname:
                res.name = new_resname

        if start > 0 or end < len(residues):
            keep = {id(r) for r in residues[start:end]}
            to_del = [i for i, r in enumerate(chain)
                      if r.entity_type == gemmi.EntityType.Polymer
                      and id(r) not in keep]
            for i in reversed(to_del):
                del chain[i]

        st.write_pdb(out_pdb)
        return None  # success

    except Exception as e:
        return str(e)


def stage_build_pdbs(classA_df: pl.DataFrame, pdb_dir: str,
                     output_pdbs_dir: str) -> tuple[int, int]:
    """Copy AlphaFold coordinates for each Class A sequence and replace
    residue names with the query sequence.

    classA_df may have multiple rows per query (one per Class A hit).
    Hits are tried in descending approx_pident order; the first hit whose
    source PDB is available and builds successfully is used.

    Returns (n_success, n_failed, timings) where timings is a list of
    elapsed seconds for each successfully built PDB.
    """
    n_ok = n_fail = 0
    timings: list[float] = []

    # group hits by query, best pident first
    from collections import defaultdict
    query_hits: dict[str, list[dict]] = defaultdict(list)
    for row in classA_df.sort('approx_pident', descending=True).iter_rows(named=True):
        query_hits[row['qseqid']].append(row)

    for qseqid, hits in query_hits.items():
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
                    print(f'  {qseqid}: built from fallback hit {rank + 1} '
                          f'({row["sseqid"]})')
                else:
                    print(f'  Written: {out_pdb}')
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

    input_basename   = Path(args.input).stem
    outdir           = f'AT_processing_{input_basename}'
    pdb_dir          = os.path.join(outdir, 'AF_pdbs')
    output_pdbs_dir  = os.path.join(outdir, 'output_pdbs_classA')
    for d in [outdir, pdb_dir, output_pdbs_dir]:
        os.makedirs(d, exist_ok=True)

    print('=' * 60)
    print('AlphaTracer 1.0  —  Class A Pipeline')
    print('=' * 60)
    print(f'  Input:     {args.input}')
    print(f'  Database:  {args.database}')
    print(f'  Output:    {outdir}/')
    print(f'  Window:    {args.window_size} aa, '
          f'>= {args.pctsim:.0f}% similarity per window')
    print()

    # ── 1. Filter ──────────────────────────────────────────────────────────────
    print('[1/4] Filtering input FASTA...')
    filtered_fasta = os.path.join(outdir, 'input_seqs_filtered.fa')
    n_in       = sum(1 for _ in SeqIO.parse(args.input, 'fasta'))
    n_filtered = stage_filter(args.input, filtered_fasta)
    while not (os.path.exists(filtered_fasta) and os.path.getsize(filtered_fasta) > 0):
        time.sleep(0.2)
    print(f'  {n_in} input → {n_filtered} passed filter')

    # ── 2. DIAMOND ────────────────────────────────────────────────────────────
    diamond_output = os.path.join(outdir, 'diamond_hits.tsv')
    if diamond_hits_valid(diamond_output):
        print(f'\n[2/4] DIAMOND: skipping (valid hits file exists: {diamond_output})')
        hits_df = pl.read_csv(diamond_output, has_header=False,
                              separator='\t', new_columns=DIAMOND_COLS)
    else:
        print('\n[2/4] DIAMOND blastp...')
        hits_df = stage_diamond(filtered_fasta, args.database, diamond_output, args.threads)
        hits_df.write_parquet(os.path.join(outdir, 'diamond_hits.pq'))

    # All hits per query above 30% pident, best first (up to 5 from DIAMOND)
    allhits_df = (
        hits_df
        .filter(pl.col('approx_pident') >= 30)
        .filter(~pl.col('full_qseq').str.contains('X'))
        .sort('approx_pident', descending=True)
    )
    n_queries = allhits_df['qseqid'].n_unique()
    allhits_df.write_parquet(os.path.join(outdir, 'allhits.pq'))
    print(f'  {len(hits_df)} raw hits → {len(allhits_df)} hits across '
          f'{n_queries} queries (>= 30% identity)')

    # ── 3. Align and classify ─────────────────────────────────────────────────
    print(f'\n[3/4] Individual NW alignment and Class A classification '
          f'(window={args.window_size} aa, pctsim={args.pctsim:.0f}%)...')
    classA_df = stage_align_and_classify(
        allhits_df, args.window_size, args.pctsim / 100
    )
    classA_df.write_parquet(os.path.join(outdir, 'classA.pq'))
    n_classA_queries = classA_df['qseqid'].n_unique()
    print(f'  {n_classA_queries} / {n_queries} queries have >= 1 Class A hit '
          f'({len(classA_df)} total Class A hit-pairs)')

    if len(classA_df) == 0:
        print('\nNo Class A sequences found. Exiting.')
        return

    # ── 4. Download PDBs ──────────────────────────────────────────────────────
    print('\n[4/4] Downloading AlphaFold reference PDBs...')
    stage_download(classA_df, pdb_dir, args.threads)

    # ── 5. Build output PDBs ──────────────────────────────────────────────────
    print('\nBuilding output PDBs...')
    n_ok, n_fail, timings = stage_build_pdbs(classA_df, pdb_dir, output_pdbs_dir)

    print()
    print('=' * 60)
    print('Class A pipeline complete.')
    print(f'  Class A sequences:  {n_classA_queries}')
    print(f'  PDBs written:       {n_ok}')
    print(f'  PDBs failed:        {n_fail}')
    if timings:
        print(f'  Time per structure: min={min(timings):.3f}s  '
              f'mean={sum(timings)/len(timings):.3f}s  '
              f'max={max(timings):.3f}s')
    print(f'  Output directory:   {output_pdbs_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
