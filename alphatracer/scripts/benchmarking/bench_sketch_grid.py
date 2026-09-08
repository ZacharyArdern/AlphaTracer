#!/usr/bin/env python3
"""
Build sketch indexes and search GTDB reps across a grid of k, alphabet, and n_hash.

Run from AT_DBs/. AFDB fasta at AFDBv6_uniref50_reps.fasta (cwd); query at ../GTDB/r232/all_reps.faa.

Stages:
  prep    — convert AFDBv6_uniref50_reps.fasta -> afdb_uniref50_reps.pq (run once on head node)
  submit  — submit one bsub job per (k, scheme, n_hash) combination
  run     — build index + search (called by each bsub job)

Usage:
  python bench_sketch_grid.py --stage prep
  python bench_sketch_grid.py --stage submit
  python bench_sketch_grid.py --stage run --k 9 --scheme murphy2000_5 --n-hash 256
"""
import argparse
import gzip
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import polars as pl
import alphatracer_sketch

AFDB_FA    = Path('AFDBv6_uniref50_reps.fasta')
AFDB_PQ    = Path('afdb_uniref50_reps.pq')
QUERY_FA   = Path('../GTDB/r232/all_reps.faa')
IDX_DIR    = Path('sketch_indexes')
HITS_DIR   = Path('sketch_hits')

K_VALUES   = list(range(6, 16))          # k = 6 .. 15
SCHEMES    = ['murphy2000_4', 'murphy2000_5', 'dayhoff1978_6', 'murphy2000_8']
N_HASHES   = [32, 64, 128, 256]
MIN_SHARED = 2
TOP_K      = 500

BUILD_THREADS  = 32
BUILD_MEM_GB   = 256    # freq table for k=15 murphy4 = 4 GB; k=15 murphy5 = 122 GB; needs headroom
SEARCH_THREADS = 32


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def job_name(k, scheme, n_hash):
    return f'k{k}_{scheme}_n{n_hash}'


def idx_path(k, scheme, n_hash):
    return IDX_DIR / f'{job_name(k, scheme, n_hash)}.sidx'


def hits_path(k, scheme, n_hash):
    return HITS_DIR / f'{job_name(k, scheme, n_hash)}.tsv.gz'


def stage_prep(chunk_size: int = 500_000):
    if AFDB_PQ.exists():
        log(f'Skipping prep — {AFDB_PQ} already exists')
        return
    if not AFDB_FA.exists():
        sys.exit(f'ERROR: {AFDB_FA} not found')
    import pyarrow as pa
    import pyarrow.parquet as pq
    log(f'Converting {AFDB_FA} -> {AFDB_PQ} (chunk_size={chunk_size:,})...')
    t0 = time.time()
    schema = pa.schema([('rep_AFDB_ID', pa.string()), ('sequence', pa.string())])
    writer = None
    total = 0
    ids, seqs = [], []
    seq_id = None
    buf = []

    def flush():
        nonlocal writer, total
        batch = pa.table({'rep_AFDB_ID': ids, 'sequence': seqs}, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(str(AFDB_PQ), schema, compression='zstd')
        writer.write_table(batch)
        total += len(ids)
        ids.clear(); seqs.clear()
        log(f'  {total:,} sequences written ({time.time()-t0:.0f}s)')

    with open(AFDB_FA) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if seq_id is not None:
                    ids.append(seq_id)
                    seqs.append(''.join(buf))
                    if len(ids) >= chunk_size:
                        flush()
                seq_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if seq_id is not None:
            ids.append(seq_id)
            seqs.append(''.join(buf))

    if ids:
        flush()
    if writer:
        writer.close()
    log(f'Done: {total:,} sequences -> {AFDB_PQ} ({time.time()-t0:.1f}s)')


def stage_submit(script_path: Path):
    IDX_DIR.mkdir(exist_ok=True)
    HITS_DIR.mkdir(exist_ok=True)

    n_submitted = 0
    for k, scheme, n_hash in product(K_VALUES, SCHEMES, N_HASHES):
        if hits_path(k, scheme, n_hash).exists():
            log(f'Skipping {job_name(k, scheme, n_hash)} — hits file exists')
            continue
        jname = job_name(k, scheme, n_hash)
        cmd = (
            f'python {script_path.resolve()} --stage run '
            f'--k {k} --scheme {scheme} --n-hash {n_hash}'
        )
        bsub = (
            f'bsub.py --threads {BUILD_THREADS} -q normal {BUILD_MEM_GB} '
            f'sketch_{jname}.log "{cmd}"'
        )
        log(f'Submitting: {bsub}')
        subprocess.run(bsub, shell=True, check=True)
        n_submitted += 1

    log(f'Submitted {n_submitted} jobs ({len(K_VALUES)*len(SCHEMES)*len(N_HASHES)} total combinations)')


def stage_run(k: int, scheme: str, n_hash: int):
    IDX_DIR.mkdir(exist_ok=True)
    HITS_DIR.mkdir(exist_ok=True)

    sidx  = idx_path(k, scheme, n_hash)
    hits  = hits_path(k, scheme, n_hash)
    label = job_name(k, scheme, n_hash)

    if not AFDB_PQ.exists():
        sys.exit(f'ERROR: {AFDB_PQ} not found')
    if not QUERY_FA.exists():
        sys.exit(f'ERROR: {QUERY_FA} not found')

    # Build index
    if not sidx.exists():
        log(f'[{label}] Building index...')
        t0 = time.perf_counter()
        n = alphatracer_sketch.build_index(
            [str(AFDB_PQ)],
            str(sidx),
            k=k,
            n_hash=n_hash,
            scheme=scheme,
            max_freq=0.001,
        )
        log(f'[{label}] Indexed {n:,} seqs in {time.perf_counter()-t0:.1f}s -> {sidx}')
    else:
        log(f'[{label}] Index exists, skipping build')

    # Load target IDs for seq_idx -> ID mapping
    log(f'[{label}] Loading target IDs from parquet...')
    target_ids = pl.scan_parquet(AFDB_PQ).select('rep_AFDB_ID').collect()['rep_AFDB_ID'].to_list()

    # Search
    log(f'[{label}] Searching {QUERY_FA}...')
    t0 = time.perf_counter()
    results = alphatracer_sketch.search_fasta(
        str(sidx),
        str(QUERY_FA),
        top_k=TOP_K,
        min_shared=MIN_SHARED,
    )
    log(f'[{label}] {len(results):,} hits in {time.perf_counter()-t0:.1f}s')

    # Write output
    log(f'[{label}] Writing hits -> {hits}')
    with gzip.open(hits, 'wt') as f:
        f.write('query_id\ttarget_id\tn_shared\tcontainment\n')
        for query_id, seq_idx, n_shared, containment in results:
            target_id = target_ids[seq_idx] if seq_idx < len(target_ids) else str(seq_idx)
            f.write(f'{query_id}\t{target_id}\t{n_shared}\t{containment:.4f}\n')

    log(f'[{label}] Done.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['prep', 'submit', 'run'])
    parser.add_argument('--k',      type=int)
    parser.add_argument('--scheme', type=str)
    parser.add_argument('--n-hash', type=int, dest='n_hash')
    args = parser.parse_args()

    if args.stage == 'prep':
        stage_prep()
    elif args.stage == 'submit':
        stage_submit(Path(__file__))
    elif args.stage == 'run':
        for a, name in [(args.k, '--k'), (args.scheme, '--scheme'), (args.n_hash, '--n-hash')]:
            if a is None:
                parser.error(f'{name} required for --stage run')
        stage_run(args.k, args.scheme, args.n_hash)


if __name__ == '__main__':
    main()
