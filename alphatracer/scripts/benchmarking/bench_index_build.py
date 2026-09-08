#!/usr/bin/env python3
"""
Benchmark sketch index build time on a sample of AFDB, then extrapolate to full DB.

Usage:
    python bench_index_build.py --pq /path/to/afdb_v6_reps.pq [--sample 10_000_000]
                                [--outdir /tmp/bench_idx]
"""
import argparse
import time
import tempfile
from pathlib import Path

import polars as pl
import alphatracer_sketch

PARAM_GRID = [
    # (k, scheme, n_hash)
    (9,  'murphy2000_5', 256),
    (8,  'murphy2000_8', 256),
    (9,  'dayhoff1978_6', 256),
    (9,  'murphy2000_4', 256),
]


def sample_parquet(pq_path: Path, n: int, out_path: Path):
    print(f'Sampling {n:,} rows from {pq_path.name}...')
    t0 = time.perf_counter()
    df = pl.scan_parquet(pq_path).head(n).collect()
    df.write_parquet(out_path)
    print(f'  Done ({time.perf_counter()-t0:.1f}s) -> {out_path}')
    return out_path


def build_and_time(pq_path: Path, sidx_path: Path, k: int, scheme: str, n_hash: int,
                   n_sample: int, n_total: int):
    label = f'k={k} {scheme} n_hash={n_hash}'
    print(f'\nBuilding index: {label}')
    t0 = time.perf_counter()
    n_seqs = alphatracer_sketch.build_index(
        [str(pq_path)],
        str(sidx_path),
        k=k,
        n_hash=n_hash,
        scheme=scheme,
        max_freq=0.001,
    )
    elapsed = time.perf_counter() - t0
    rate = n_sample / elapsed
    extrap_h = n_total / rate / 3600
    size_mb = sidx_path.stat().st_size / 1e6

    print(f'  {n_seqs:,} seqs indexed in {elapsed:.1f}s  ({rate/1e6:.2f}M seqs/s)')
    print(f'  Index size: {size_mb:.1f} MB')
    print(f'  Extrapolated full DB ({n_total/1e6:.0f}M seqs): ~{extrap_h:.2f}h')
    return {'label': label, 'elapsed_s': elapsed, 'rate_Mseqs_s': rate/1e6,
            'extrap_h': extrap_h, 'size_mb': size_mb}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pq', default='/Users/zacharyardern/Science/Data/AFDB/afdb_v6_reps.pq',
                        type=Path)
    parser.add_argument('--sample', type=int, default=10_000_000)
    parser.add_argument('--outdir', type=Path, default=Path(tempfile.mkdtemp(prefix='bench_idx_')))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_total = pl.scan_parquet(args.pq).select(pl.len()).collect()[0, 0]
    print(f'Full DB: {n_total:,} sequences')

    sample_pq = args.outdir / 'sample.parquet'
    if not sample_pq.exists():
        sample_parquet(args.pq, args.sample, sample_pq)

    results = []
    for k, scheme, n_hash in PARAM_GRID:
        sidx = args.outdir / f'idx_k{k}_{scheme}_n{n_hash}.sidx'
        r = build_and_time(sample_pq, sidx, k, scheme, n_hash, args.sample, n_total)
        results.append(r)

    print('\n\n=== Summary ===')
    print(f'{"Config":<35} {"Rate (M/s)":>10} {"Sample (s)":>11} {"Full DB (h)":>12} {"Size (MB)":>10}')
    print('-' * 82)
    for r in results:
        print(f'{r["label"]:<35} {r["rate_Mseqs_s"]:>10.2f} {r["elapsed_s"]:>11.1f} '
              f'{r["extrap_h"]:>12.2f} {r["size_mb"]:>10.1f}')


if __name__ == '__main__':
    main()
