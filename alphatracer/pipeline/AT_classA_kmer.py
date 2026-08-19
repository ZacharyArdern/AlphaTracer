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

import asyncio
import os, re, subprocess, sys, time, argparse, pickle, threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import aiohttp
import polars as pl
import parasail
import gemmi
import duckdb
from alphatracer.utils.afdb_fetch import (
    AFDB_VERSION, get_afdb_id, afdb_local_pdb, is_valid_pdb,
    fetch_afdb_pdbs, parse_fasta,
)
from alphatracer.utils.esm_atlas_fetch import esm_local_pdb, _ESM_DIR, fetch_esm_structures

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
_ID_COL            = None   # name of the ID column in REPS_PQ
_HAS_ANN           = True   # False for DBs without function/family/group_size/n_reps columns
_HAS_DB_TYPE       = False  # True for merged AFDB+ESMAtlas DBs with db_type column; auto-detected
_HAS_SEQ_IDX       = False  # True when parquet has seq_idx integer column (enables DuckDB min/max pruning)
_IS_STANDALONE_ESM = False  # True for standalone ESMAtlas DBs (header+sequence only, no db_type col)

# _ESM_DIR imported from alphatracer.utils.esm_atlas_fetch


def _configure_db(sketch_db: str | None = None) -> None:
    """Probe parquet schema and set _ID_COL / _HAS_ANN / _HAS_DB_TYPE.

    If sketch_db is given, also redirects REPS_PQ / SIDX_CACHE / ANN_CACHE to
    that path.  Called unconditionally in main() so merged AFDB+ESMAtlas parquets
    are auto-detected even on the default path.
    """
    global REPS_PQ, SIDX_CACHE, ANN_CACHE, _ID_COL, _HAS_ANN, _HAS_DB_TYPE, _HAS_SEQ_IDX, _IS_STANDALONE_ESM
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
    names = pl.read_parquet(REPS_PQ, n_rows=0).columns
    if "rep_AFDB_ID" in names:
        _ID_COL = "rep_AFDB_ID"
    elif "AFDB_ID" in names:
        _ID_COL = "AFDB_ID"
    elif "header" in names:
        _ID_COL = "header"
    else:
        sys.exit(f"[FATAL] parquet has no recognised ID column "
                 f"(expected rep_AFDB_ID, AFDB_ID, or header). Found: {names}")
    _HAS_ANN           = all(c in names for c in ("function", "family", "group_size", "n_reps"))
    _HAS_DB_TYPE       = "db_type" in names
    _HAS_SEQ_IDX       = "seq_idx" in names
    # Standalone ESMAtlas DB: just header + sequence, no db_type column.
    # db_type/protein_hash columns are injected into hits_df post-search.
    _IS_STANDALONE_ESM = (_ID_COL == "header" and not _HAS_DB_TYPE and not _HAS_ANN)
    if _IS_STANDALONE_ESM:
        print("[DB] Standalone ESMAtlas DB detected — will tag hits as db_type='esm_atlas'", flush=True)

# dayhoff_sketch source is bundled in the package; binaries are compiled into a user cache dir.
_SKETCH_RS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dayhoff_sketch')
_SKETCH_RS_CARGO_TARGET = os.path.join(
    os.environ.get('AT_CACHE_DIR', os.path.expanduser("~/.cache/alphatracer")),
    'dayhoff_sketch',
)
_SKETCH_RS_BIN = os.path.join(_SKETCH_RS_CARGO_TARGET, 'release')

# AFDB_VERSION, get_afdb_id, afdb_local_pdb, is_valid_pdb, esm_local_pdb
# imported from alphatracer.utils.afdb_fetch

K            = 9
MAX_FREQ     = 0.001
MIN_SHARED   = 2
N_HASH_SEARCH = 0  # 0 = use index n_hash; set from --n-hash-search in main()

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
    p.add_argument('--top-k', type=int, default=100,
                   help='Hits per query from kmer search (default: 100)')
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
    p.add_argument('--sketch-db', default=None, nargs='+', metavar='PATH',
                   help='Path(s) to sketch database parquet(s) (e.g. AFDB, ESMAtlas). '
                        'Each must have columns [<id>, sequence] where <id> is one of '
                        'rep_AFDB_ID / AFDB_ID / header. Multiple paths are searched '
                        'sequentially and hits merged. Overrides AT_AFDB_DIR defaults.')
    p.add_argument('--n-hash-search', type=int, default=0, metavar='N',
                   help='Number of query hashes used during search (default: 0 = index n_hash). '
                        'Set higher than the index n_hash for containment-style search.')
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


# get_afdb_id, afdb_local_pdb, is_valid_pdb, _ESM_DIR, esm_local_pdb,
# fetch_afdb_pdbs imported from alphatracer.utils.afdb_fetch
# fetch_esm_structures imported from alphatracer.utils.esm_atlas_fetch


def _fetch_pdb(afdb_id, pdb_dir):
    """Synchronous single-file fetch (on-demand fallback)."""
    from alphatracer.utils.afdb_fetch import _fetch_afdb_pdbs_async
    return asyncio.run(_fetch_afdb_pdbs_async([afdb_id], pdb_dir))[afdb_id]


# ── Stage 1: filter FASTA ──────────────────────────────────────────────────────

def stage_filter(input_path, output_path):
    """Write filtered FASTA (no 'X', <= 2000 aa). Returns (n_in, n_out, seq_dict)."""
    filtered = []
    seq_dict = {}
    n_in = 0
    print("  Parsing input FASTA...", flush=True)
    records = list(parse_fasta(input_path))
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
    print("  dayhoff_sketch binaries not found — building with cargo (one-time, ~30s)...", flush=True)
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
    print("  dayhoff_sketch binaries built and cached.", flush=True)
    return index_seqs, search


def _build_sidx(index_seqs_bin):
    """Build V2 .sidx inverted index directly from reps.pq (one step, no sketch parquet).

    The index-seqs binary requires the ID column to be named 'AFDB_ID' or 'rep_AFDB_ID'.
    If the source parquet uses a different name (e.g. 'header' for ESMAtlas), a renamed
    temp parquet is streamed first, then cleaned up after indexing.
    """
    global _ID_COL
    n = pl.scan_parquet(REPS_PQ).select(pl.len()).collect().item()
    print(f"  Building V2 inverted index from {n:,} sequences (runs once, cached)...", flush=True)
    t0 = time.time()

    id_col = _ID_COL or "rep_AFDB_ID"
    if id_col in ("rep_AFDB_ID", "AFDB_ID"):
        src_pq = REPS_PQ
        tmp_pq = None
    else:
        # Write a renamed copy so index-seqs sees the expected column name.
        tmp_pq = REPS_PQ.rsplit('.', 1)[0] + "_idrenamed_tmp.pq"
        print(f"  Writing renamed parquet ('{id_col}' → 'AFDB_ID') → {tmp_pq}", flush=True)
        pl.read_parquet(REPS_PQ).rename({id_col: 'AFDB_ID'}).write_parquet(
            tmp_pq, compression='zstd'
        )
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


def _fetch_by_row_indices(pq_path: str, row_indices: set, cols: list) -> dict:
    """Fetch parquet rows by row index via polars scan with row_index filter.

    Returns {row_idx: {col: value}} for each requested index.
    Used only for parquets without a seq_idx column (e.g. ESM standalone).
    """
    idx_list = sorted(row_indices)
    df = (
        pl.scan_parquet(pq_path)
        .with_row_index("_ri")
        .filter(pl.col("_ri").is_in(idx_list))
        .select(["_ri"] + cols)
        .collect()
    )
    return {
        row[0]: {col: row[i + 1] for i, col in enumerate(cols)}
        for row in df.iter_rows()
    }


def stage_kmer_search(filtered_fasta, query_seq_dict, outfile, top_k):
    """
    Run Rust kmer sketch search against the AFDB rep index.

    Returns a polars DataFrame with columns matching what stage_align_and_classify expects:
      containment_value, sseqid, qseqid, evalue, slen, qlen,
      full_qseq, full_sseq, function, family, group_size, n_reps
    """
    index_seqs_bin, search_bin = _ensure_binaries()

    if not os.path.exists(SIDX_CACHE):
        _build_sidx(index_seqs_bin)
    else:
        print(f"  Using cached inverted index: {SIDX_CACHE}", flush=True)

    print(f"  Running Rust search (top_k={top_k})...", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [search_bin, SIDX_CACHE, filtered_fasta,
         str(top_k), str(MIN_SHARED), str(K), str(N_HASH_SEARCH)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    # Stream stderr live (progress bar) while collecting stdout
    stdout_lines = []
    def _drain_stderr():
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    stdout_lines = proc.stdout.read().splitlines()
    proc.wait()
    stderr_thread.join()
    if proc.returncode != 0:
        sys.exit(1)
    lines = stdout_lines
    header   = lines[0] if lines else ""
    raw_hits = lines[1:] if lines else []
    is_row_idx = "row_idx" in header  # v3 sidx: targets are parquet row indices
    print(f"  {len(raw_hits)} hits in {time.time()-t0:.1f}s", flush=True)

    _extra_schema = {
        'db_type': pl.String, 'afdb_id': pl.String,
        'protein_hash': pl.String, 'fragment_id': pl.Int32, 'frag_row': pl.Int32,
    } if (_HAS_DB_TYPE or _IS_STANDALONE_ESM) else {}

    if not raw_hits:
        return pl.DataFrame(schema={
            'containment_value': pl.Float64, 'sseqid': pl.String, 'qseqid': pl.String,
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
        query, target, shared, containment_value = parts[0], parts[1], int(parts[2]), float(parts[3])
        parsed.append((query, target, shared, containment_value))
        targets.add(target)

    # ── Progressive annotation cache ──────────────────────────────────────────
    # Load what we already know; fetch only genuinely new targets.
    id_col  = _ID_COL or "rep_AFDB_ID"
    ann_pkl = _load_ann_cache()
    missing = targets - ann_pkl.keys()
    print(f"  Annotations: {len(targets)} targets "
          f"({len(targets) - len(missing)} from cache, {len(missing)} new)...", flush=True)
    if missing:
        t_ann = time.time()
        if is_row_idx:
            missing_idx = [int(t) for t in missing]
            if _HAS_SEQ_IDX:
                # DuckDB with seq_idx integer column: min/max statistics allow row-group
                # skipping in C++, faster than Python-level row-group reads.
                db_rows = duckdb.connect().execute(
                    f"SELECT seq_idx, {id_col}, sequence FROM read_parquet('{REPS_PQ}')"
                    f" WHERE seq_idx IN (SELECT unnest(?))",
                    [missing_idx]
                ).fetchall()
                for seq_idx_val, afdb_id_val, seq in db_rows:
                    ann_pkl[str(seq_idx_val)] = (afdb_id_val or "", "", 0, 0, seq)
            else:
                # Fallback: pyarrow row-group reads (fast for small batches).
                rows = _fetch_by_row_indices(REPS_PQ, set(missing_idx), [id_col, 'sequence'])
                for row_idx_int, data in rows.items():
                    ann_pkl[str(row_idx_int)] = (data.get(id_col, ""), "", 0, 0, data['sequence'])
            print(f"  Row-idx fetch ({'seq_idx' if _HAS_SEQ_IDX else 'row-group'}): "
                  f"{len(missing_idx)} rows in {time.time()-t_ann:.1f}s", flush=True)
        else:
            if _HAS_DB_TYPE:
                cols = [id_col, 'db_type', 'afdb_id', 'protein_hash',
                        'fragment_id', 'frag_row', 'function', 'family',
                        'group_size', 'n_reps', 'sequence']
            elif _HAS_ANN:
                cols = [id_col, 'function', 'family', 'group_size', 'n_reps', 'sequence']
            else:
                cols = [id_col, 'sequence']
            df = (pl.scan_parquet(REPS_PQ)
                    .filter(pl.col(id_col).is_in(list(missing)))
                    .select(cols)
                    .collect())
            for row in df.iter_rows():
                if _HAS_DB_TYPE or _HAS_ANN:
                    ann_pkl[row[0]] = row[1:]
                else:
                    ann_pkl[row[0]] = ("", "", 0, 0, row[1])
            print(f"  Polars query: {len(missing)} new targets in {time.time()-t_ann:.1f}s  "
                  f"(cache now {len(ann_pkl):,} entries)", flush=True)
        _save_ann_cache(ann_pkl)

    if _HAS_DB_TYPE:
        # ann_pkl value: (db_type, afdb_id, protein_hash, frag_id, frag_row, func, fam, gs, nr, seq)
        ann_map  = {t: ann_pkl[t][5:9] for t in targets if t in ann_pkl}   # func,fam,gs,nr
        seq_map  = {t: ann_pkl[t][9]   for t in targets if t in ann_pkl}
        meta_map = {t: ann_pkl[t][:5]  for t in targets if t in ann_pkl}   # db_type..frag_row
    else:
        if is_row_idx:
            # Position 0 holds afdb_id (not func); annotations are always blank for row_idx entries.
            ann_map = {t: ("", "", 0, 0) for t in targets if t in ann_pkl}
        else:
            ann_map = {t: ann_pkl[t][:4] for t in targets if t in ann_pkl}
        seq_map  = {t: ann_pkl[t][4]  for t in targets if t in ann_pkl}
        meta_map = {}

    # For v3 sidx: build row_idx → afdb_id map; heal stale cache entries that lack it.
    _row_idx_to_afdb = {}
    if is_row_idx:
        id_col_local = _ID_COL or "rep_AFDB_ID"
        stale = [int(t) for t in targets if t in ann_pkl and not ann_pkl[t][0]]
        if stale:
            if _HAS_SEQ_IDX:
                stale_db = duckdb.connect().execute(
                    f"SELECT seq_idx, {id_col_local} FROM read_parquet('{REPS_PQ}')"
                    f" WHERE seq_idx IN (SELECT unnest(?))", [stale]
                ).fetchall()
                for row_idx_int, afdb_id_val in stale_db:
                    existing = ann_pkl[str(row_idx_int)]
                    ann_pkl[str(row_idx_int)] = (afdb_id_val or "",) + existing[1:]
            else:
                stale_rows = _fetch_by_row_indices(REPS_PQ, set(stale), [id_col_local])
                for row_idx_int, data in stale_rows.items():
                    existing = ann_pkl[str(row_idx_int)]
                    ann_pkl[str(row_idx_int)] = (data.get(id_col_local, ""),) + existing[1:]
            _save_ann_cache(ann_pkl)
        _row_idx_to_afdb = {t: ann_pkl[t][0] for t in targets if t in ann_pkl and ann_pkl[t][0]}

    # Save raw search output for reference
    with open(outfile, 'w') as f:
        f.write("query\ttarget\tshared\tcontainment_value\tfunction\tfamily\tsequence\n")
        for query, target, shared, containment_value in parsed:
            func, fam, gs, nr = ann_map.get(target, ("", "", 0, 0))
            seq = seq_map.get(target, "")
            f.write(f"{query}\t{target}\t{shared}\t{containment_value:.4f}\t{func}\t{fam}\t{seq}\n")

    rows = []
    for query, target, shared, containment_value in parsed:
        func, fam, gs, nr = ann_map.get(target, ("", "", 0, 0))
        full_qseq = query_seq_dict.get(query, "")
        full_sseq = seq_map.get(target, "")
        if not full_qseq or not full_sseq:
            continue
        row = {
            'containment_value':       containment_value,
            'sseqid':        _row_idx_to_afdb.get(target, target),
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

    df = pl.DataFrame(rows)

    # Inject ESM metadata columns for standalone ESMAtlas DB (no db_type in parquet schema).
    # sseqid format: <32hex>|<sha256/MGYP>|<spire/mgy> — protein_hash is the 32-char prefix.
    if _IS_STANDALONE_ESM and len(df) > 0:
        df = df.with_columns([
            pl.lit('esm_atlas').alias('db_type'),
            pl.lit('').alias('afdb_id'),
            pl.col('sseqid').str.split('|').list.first().alias('protein_hash'),
            pl.lit(-1).cast(pl.Int32).alias('fragment_id'),
            pl.lit(-1).cast(pl.Int32).alias('frag_row'),
        ])
    return df


# ── Stage 3: align and classify ────────────────────────────────────────────────

def stage_align_and_classify(hits_df, window, threshold, threads=4):
    """Align and classify with early exit: candidates are processed in descending
    containment_value order per query; alignment stops as soon as a Class A hit is found.
    Parallelises across queries (not pairs), so threads are fully utilised."""
    total = len(hits_df)
    n_queries = hits_df['qseqid'].n_unique()
    print(f"  Aligning {total} candidates for {n_queries} queries "
          f"(early-exit, threads={threads})...", flush=True)

    # Group rows by query, already sorted descending by containment_value from the search binary.
    groups: dict[str, list] = {}
    for row in hits_df.sort('containment_value', descending=True).iter_rows(named=True):
        groups.setdefault(row['qseqid'], []).append(row)

    def _align_query(query_rows: list) -> dict | None:
        """Align candidates for one query in ranked order; return first Class A hit."""
        for row in query_rows:
            result = align_nw(row['qseqid'], row['sseqid'],
                              row['full_qseq'], row['full_sseq'])
            if result is None:
                continue
            _, _, qseq_alg, sseq_alg, alg_comp = result
            qseq_alg = fix_hanging_group_letters(qseq_alg)
            sseq_alg = fix_hanging_group_letters(sseq_alg)
            alg_comp = fix_hanging_group_letters(alg_comp)
            if is_classA(qseq_alg, sseq_alg, alg_comp, window, threshold):
                return {**row, 'qseq_alg': qseq_alg,
                        'sseq_alg': sseq_alg, 'alg_comp': alg_comp}
        return None

    classA_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for result in ex.map(_align_query, groups.values()):
            done += 1
            if result:
                classA_rows.append(result)
            if done % 100 == 0 or done == n_queries:
                print(f'  Processed {done}/{n_queries} queries, '
                      f'{len(classA_rows)} Class A so far...', end='\r', flush=True)
    print()

    if not classA_rows:
        return pl.DataFrame(schema={
            **{c: hits_df.schema[c] for c in hits_df.columns},
            'qseq_alg': pl.String, 'sseq_alg': pl.String, 'alg_comp': pl.String,
        })

    return pl.DataFrame(classA_rows)


# ── Stage 4: download PDBs ─────────────────────────────────────────────────────

def stage_download(classA_df, pdb_dir, threads):
    top_sseqids = (
        classA_df
        .sort('containment_value', descending=True)
        .group_by('qseqid', maintain_order=True)
        .agg(pl.col('sseqid').first())
        ['sseqid'].to_list()
    )
    afdb_ids = {get_afdb_id(sid) for sid in top_sseqids if get_afdb_id(sid)}
    print(f'  {len(afdb_ids)} unique AlphaFold accession(s) to download (v{AFDB_VERSION})',
          flush=True)
    fetch_afdb_pdbs(afdb_ids, pdb_dir)


# ── Stage 4+5: pipelined download and build ────────────────────────────────────

def stage_download_and_build(classA_df, pdb_dir, output_pdbs_dir, threads):
    """Download reference PDBs and build output PDBs in a pipelined fashion.

    Each query's output PDB is built immediately after its primary reference PDB
    download completes, so builds overlap with remaining downloads.
    ESM Atlas structures are fetched from Lance S3 before the AFDB download loop.
    """
    # ── Pre-fetch ESM Atlas structures in background (concurrent with AFDB downloads)
    _esm_fetch_thread = None
    if 'db_type' in classA_df.columns:
        esm_df = classA_df.filter(pl.col('db_type') == 'esm_atlas')
        if len(esm_df) > 0:
            n_esm = esm_df['protein_hash'].n_unique()
            print(f'  Pre-fetching {n_esm} unique ESM Atlas structure(s) (background)...', flush=True)
            _esm_fetch_thread = threading.Thread(
                target=fetch_esm_structures,
                args=(esm_df.unique(subset=['protein_hash']).to_dicts(), pdb_dir),
                kwargs={'n_workers': 16}, daemon=True)
            _esm_fetch_thread.start()

    query_hits = defaultdict(list)
    for row in classA_df.sort('containment_value', descending=True).iter_rows(named=True):
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

    # ── download all primary PDBs ──────────────────────────────────────────
    results = fetch_afdb_pdbs(all_aids, pdb_dir)

    # ── trigger builds for successfully downloaded primary hits ────────────
    for aid, r in results.items():
        if not r.startswith('fail'):
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
        src_pdb = esm_local_pdb(protein_hash, pdb_dir)
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
    for row in classA_df.sort('containment_value', descending=True).iter_rows(named=True):
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

    global N_HASH_SEARCH
    N_HASH_SEARCH = args.n_hash_search

    # Resolve DB paths: list of paths, or [None] meaning default AFDB.
    _db_paths = args.sketch_db if args.sketch_db else [None]

    # Configure with the first DB now so globals are set for --download-build-only.
    _configure_db(_db_paths[0])

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
    if len(_db_paths) == 1:
        print(f'  Index:       {SIDX_CACHE}')
    else:
        print(f'  Databases:   {len(_db_paths)} (searched sequentially, hits merged)')
        for _p in _db_paths:
            print(f'               {_p or "(default AFDB)"}')
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

    _all_hits: list[pl.DataFrame] = []
    _classA_ids: set[str] = set()   # queries confirmed Class A after each DB
    _current_fasta   = filtered_fasta
    _current_seq_dict = query_seq_dict

    for _db_idx, _db_path in enumerate(_db_paths):
        if len(_db_paths) > 1:
            _label = _db_path or '(default AFDB)'
            print(f'\n  DB {_db_idx + 1}/{len(_db_paths)}: {_label}'
                  + (f'  ({len(_current_seq_dict)} queries remaining)' if _db_idx > 0 else ''))
            _configure_db(_db_path)
        _hits_part = stage_kmer_search(_current_fasta, _current_seq_dict, kmer_out, args.top_k)
        if 'db_type' not in _hits_part.columns and len(_hits_part) > 0:
            _hits_part = _hits_part.with_columns([
                pl.lit('afdb').alias('db_type'),
                pl.lit('').alias('afdb_id'),
                pl.lit('').alias('protein_hash'),
                pl.lit(-1).cast(pl.Int32).alias('fragment_id'),
                pl.lit(-1).cast(pl.Int32).alias('frag_row'),
            ])
        _all_hits.append(_hits_part)

        # After each DB except the last: classify hits so far, exclude confirmed
        # Class A queries from subsequent DB searches.
        if _db_idx < len(_db_paths) - 1 and len(_hits_part) > 0:
            _hits_so_far = pl.concat(_all_hits, how='diagonal_relaxed').filter(
                pl.col('full_qseq').str.len_chars() > 0,
                pl.col('full_sseq').str.len_chars() > 0,
            )
            print(f'  Classifying hits from DB {_db_idx + 1} to skip confirmed Class A queries...')
            _t_cls = time.time()
            _interim_classA = stage_align_and_classify(
                _hits_so_far, args.window_size, args.pctsim / 100, args.threads)
            _new_classA = set(_interim_classA['qseqid'].to_list()) - _classA_ids
            _classA_ids |= _new_classA
            print(f'  {len(_classA_ids)} Class A confirmed so far — '
                  f'skipping from remaining DB searches  [{time.time()-_t_cls:.1f}s]')

            # Write reduced FASTA for next DB search.
            _current_seq_dict = {k: v for k, v in query_seq_dict.items()
                                 if k not in _classA_ids}
            _current_fasta = os.path.join(outdir, 'input_seqs_remaining.fa')
            with open(_current_fasta, 'w') as _ff:
                _ff.write('\n'.join(f'>{k}\n{v}' for k, v in _current_seq_dict.items()))
            print(f'  {len(_current_seq_dict)} queries remaining for next DB')

    if len(_all_hits) == 1:
        hits_df = _all_hits[0]
    else:
        hits_df = pl.concat(_all_hits, how='diagonal_relaxed')
        hits_df = (hits_df
                   .sort('containment_value', descending=True)
                   .unique(subset=['qseqid', 'sseqid'], keep='first'))
        print(f'\n  Merged hits from {len(_db_paths)} databases: {len(hits_df)} unique (qseqid, sseqid) pairs')

    n_queries_hit = hits_df['qseqid'].n_unique() if len(hits_df) > 0 else 0
    hits_df.write_parquet(os.path.join(outdir, 'kmer_hits.pq'))
    hits_df.write_parquet(os.path.join(outdir, 'allhits.pq'))
    print(f'  {len(hits_df)} hits across {n_queries_hit} queries  [{time.time()-t2:.1f}s]')
    print(f'  Raw hits saved to: {kmer_out}')

    if len(hits_df) == 0:
        print('\nNo kmer hits found. Exiting.')
        return

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

    _show_by_db = _HAS_DB_TYPE or len(_db_paths) > 1
    print()
    print('=' * 60)
    print('Class A pipeline complete.')
    print(f'  Class A sequences:  {n_classA_queries}')
    print(f'  PDBs written:       {n_ok}')
    if _show_by_db:
        for _dbt, _cnt in sorted(ok_by_db.items()):
            print(f'    {_dbt}:  {_cnt}')
    print(f'  PDBs failed:        {n_fail}')
    if _show_by_db and n_fail:
        for _dbt, _cnt in sorted(fail_by_db.items()):
            if _cnt:
                print(f'    {_dbt}:  {_cnt}')
    if timings:
        print(f'  Time per structure: min={min(timings):.3f}s  '
              f'mean={sum(timings)/len(timings):.3f}s  '
              f'max={max(timings):.3f}s')
    print(f'  Total runtime:      {time.time()-t_total:.1f}s')
    print(f'  Output directory:   {output_pdbs_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
