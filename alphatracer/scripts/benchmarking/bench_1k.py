#!/usr/bin/env python3
"""
Benchmark 1000 random queries: all sketch param combos (top_k=500) + diamond --fast,
evaluated against diamond ultra-sensitive top-500 ground truth.

Run from AT_DBs/.

Stages (run in order):
  prep    — extract 1000 random queries from all_reps.faa + build ground truth parquet
  sketch  — search all existing .sidx indexes with top_k=500 (reuses built indexes)
  diamond — submit bsub diamond --fast job against afdb_uniref50_reps_db.dmnd
  eval    — compare all methods against ground truth, write eval_1k_results.tsv

Usage:
  python bench_1k.py --stage prep
  python bench_1k.py --stage sketch
  python bench_1k.py --stage diamond
  python bench_1k.py --stage eval
"""
import argparse
import re
import random
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

QUERY_FA      = Path('../GTDB/r232/all_reps.faa')
AFDB_PQ       = Path('afdb_uniref50_reps.pq')
AFDB_DMND     = Path('afdb_uniref50_reps_db.dmnd')
DIAMOND_TSV   = Path('afdb_hits_merged.tsv')
IDX_DIR       = Path('sketch_indexes')

QUERY_1K      = Path('query_1k.faa')
GT_PQ         = Path('gt_1k.pq')
HITS_1K_DIR   = Path('hits_1k')
DMND_FAST_TSV = Path('diamond_fast_1k.tsv')
OUT_TSV       = Path('eval_1k_results.tsv')

N_QUERIES     = 1000
SEED          = 42
TOP_K         = 500
MIN_SHARED    = 2
EVALUE_MAX    = 1e-10
MIN_ALN_LEN   = 40
PIDENT_MIN    = 30.0
BIN_LABELS    = ['30-40%', '40-50%', '50-60%', '60%+']

DMND_THREADS  = 32
DMND_MEM_GB   = 64


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ── prep ──────────────────────────────────────────────────────────────────────

def parse_fasta(path: Path) -> dict:
    seqs, seq_id, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if seq_id:
                    seqs[seq_id] = ''.join(buf)
                seq_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if seq_id:
        seqs[seq_id] = ''.join(buf)
    return seqs


def stage_prep():
    if not QUERY_1K.exists():
        log(f'Parsing {QUERY_FA}...')
        seqs = parse_fasta(QUERY_FA)
        log(f'  {len(seqs):,} sequences total')
        rng = random.Random(SEED)
        chosen = rng.sample(sorted(seqs.keys()), N_QUERIES)
        with open(QUERY_1K, 'w') as f:
            for sid in chosen:
                f.write(f'>{sid}\n{seqs[sid]}\n')
        log(f'  Written {N_QUERIES} queries -> {QUERY_1K}')
    else:
        log(f'Skipping query extraction — {QUERY_1K} exists')

    if GT_PQ.exists():
        log(f'Skipping ground truth prep — {GT_PQ} exists')
        return

    chosen_ids = {line[1:].split()[0].strip() for line in open(QUERY_1K) if line.startswith('>')}
    log(f'Building ground truth for {len(chosen_ids):,} queries from {DIAMOND_TSV}...')
    t0 = time.time()

    (
        pl.scan_csv(DIAMOND_TSV, separator='\t', has_header=False,
                    new_columns=['query', 'target', 'pident', 'length', 'qlen', 'evalue'],
                    schema_overrides={'pident': pl.Float32, 'length': pl.Int32,
                                      'evalue': pl.Float64})
        .filter(pl.col('query').is_in(list(chosen_ids)))
        .filter(pl.col('evalue') <= EVALUE_MAX)
        .filter(pl.col('pident') >= PIDENT_MIN)
        .filter(pl.col('length') >= MIN_ALN_LEN)
        .select(['query', 'target', 'pident', 'evalue'])
        .with_columns(
            pl.when(pl.col('pident').is_between(30, 40, closed='left')).then(pl.lit('30-40%'))
            .when(pl.col('pident').is_between(40, 50, closed='left')).then(pl.lit('40-50%'))
            .when(pl.col('pident').is_between(50, 60, closed='left')).then(pl.lit('50-60%'))
            .otherwise(pl.lit('60%+')).alias('pident_bin')
        )
        .sort(['query', 'evalue'])
        .with_columns(pl.col('evalue').rank('ordinal').over('query').alias('rank'))
        .filter(pl.col('rank') <= TOP_K)
        .drop('rank')
        .sink_parquet(GT_PQ, compression='zstd')
    )
    log(f'Done in {time.time()-t0:.1f}s -> {GT_PQ}')
    counts = pl.scan_parquet(GT_PQ).group_by('pident_bin').agg(pl.len().alias('n')).collect().sort('pident_bin')
    for row in counts.to_dicts():
        log(f'  {row["pident_bin"]}: {row["n"]:,}')


# ── sketch ────────────────────────────────────────────────────────────────────

def stage_sketch():
    import alphatracer_sketch
    HITS_1K_DIR.mkdir(exist_ok=True)

    if not AFDB_PQ.exists():
        sys.exit(f'ERROR: {AFDB_PQ} not found')
    log('Loading target IDs from parquet...')
    target_ids = pl.scan_parquet(AFDB_PQ).select('rep_AFDB_ID').collect()['rep_AFDB_ID'].to_list()

    sidx_files = sorted(IDX_DIR.glob('*.sidx'))
    log(f'Searching {len(sidx_files)} indexes against {QUERY_1K}...')

    for i, sidx in enumerate(sidx_files):
        out = HITS_1K_DIR / (sidx.stem + '.tsv')
        if out.exists():
            log(f'  [{i+1}/{len(sidx_files)}] Skipping {sidx.stem}')
            continue
        t0 = time.perf_counter()
        results = alphatracer_sketch.search_fasta(
            str(sidx), str(QUERY_1K), top_k=TOP_K, min_shared=MIN_SHARED)
        elapsed = time.perf_counter() - t0
        with open(out, 'w') as f:
            f.write('query_id\ttarget_id\tn_shared\tcontainment\n')
            for query_id, seq_idx, n_shared, containment in results:
                tid = target_ids[seq_idx] if seq_idx < len(target_ids) else str(seq_idx)
                f.write(f'{query_id}\t{tid}\t{n_shared}\t{containment:.4f}\n')
        log(f'  [{i+1}/{len(sidx_files)}] {sidx.stem}: {len(results):,} hits in {elapsed:.1f}s')


# ── diamond ───────────────────────────────────────────────────────────────────

def stage_diamond():
    if DMND_FAST_TSV.exists():
        log(f'Skipping diamond fast — {DMND_FAST_TSV} exists')
        return
    if not AFDB_DMND.exists():
        sys.exit(f'ERROR: {AFDB_DMND} not found')
    cmd = (
        f'diamond blastp -q {QUERY_1K} -d {AFDB_DMND} -o {DMND_FAST_TSV} '
        f'--outfmt 6 qseqid sseqid pident length qlen evalue '
        f'--fast --max-target-seqs {TOP_K} --evalue {EVALUE_MAX} --min-score 0 '
        f'--threads {DMND_THREADS}'
    )
    bsub = f'bsub.py --threads {DMND_THREADS} -q normal {DMND_MEM_GB} diamond_fast_1k.log "{cmd}"'
    log(f'Submitting: {bsub}')
    subprocess.run(bsub, shell=True, check=True)


# ── eval ──────────────────────────────────────────────────────────────────────

def parse_job_name(stem: str) -> tuple | None:
    m = re.search(r'k(\d+)_(murphy\w+|dayhoff\w+)_n(\d+)', stem)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def compute_metrics(hits: pl.DataFrame, gt_pairs: set,
                    gt_df: pl.DataFrame, meta: dict) -> dict:
    n_hits = hits.height
    hit_set = set(zip(hits['query_id'].to_list(), hits['target_id'].to_list()))
    n_tp_total = len(hit_set & gt_pairs)
    precision = n_tp_total / n_hits if n_hits > 0 else 0.0
    row = {**meta, 'n_hits': n_hits, 'precision_overall': round(precision, 4)}
    for label in BIN_LABELS:
        bin_df = gt_df.filter(pl.col('pident_bin') == label)
        n_gt = bin_df.height
        if n_gt == 0:
            row[f'recall_{label}'] = None
            row[f'n_gt_{label}'] = 0
            row[f'n_tp_{label}'] = 0
        else:
            bin_pairs = set(zip(bin_df['query'].to_list(), bin_df['target'].to_list()))
            n_tp = len(hit_set & bin_pairs)
            row[f'recall_{label}'] = round(n_tp / n_gt, 4)
            row[f'n_gt_{label}'] = n_gt
            row[f'n_tp_{label}'] = n_tp
    return row


def stage_eval():
    if not GT_PQ.exists():
        sys.exit(f'ERROR: {GT_PQ} not found — run --stage prep first')

    gt_df = pl.read_parquet(GT_PQ)
    gt_pairs = set(zip(gt_df['query'].to_list(), gt_df['target'].to_list()))
    log(f'Ground truth: {len(gt_pairs):,} pairs, {gt_df["query"].n_unique():,} queries')

    results = []

    # Sketch
    sketch_files = sorted(HITS_1K_DIR.glob('*.tsv'))
    log(f'Evaluating {len(sketch_files)} sketch result files...')
    for path in sketch_files:
        params = parse_job_name(path.stem)
        if params is None:
            continue
        k, scheme, n_hash = params
        try:
            hits = pl.read_csv(path, separator='\t', has_header=True,
                               schema_overrides={'n_shared': pl.UInt32, 'containment': pl.Float32})
        except Exception as e:
            log(f'  WARNING: {path.name}: {e}')
            continue
        if hits.is_empty():
            continue
        row = compute_metrics(hits.select(['query_id', 'target_id']), gt_pairs, gt_df,
                              {'method': 'sketch', 'k': k, 'scheme': scheme, 'n_hash': n_hash})
        results.append(row)

    # Diamond fast
    if DMND_FAST_TSV.exists():
        log('Evaluating diamond fast...')
        dmnd = (
            pl.read_csv(DMND_FAST_TSV, separator='\t', has_header=False,
                        new_columns=['query_id', 'target_id', 'pident', 'length', 'qlen', 'evalue'],
                        schema_overrides={'pident': pl.Float32, 'length': pl.Int32,
                                          'evalue': pl.Float64})
            .filter(pl.col('evalue') <= EVALUE_MAX)
            .filter(pl.col('pident') >= PIDENT_MIN)
            .filter(pl.col('length') >= MIN_ALN_LEN)
            .select(['query_id', 'target_id'])
        )
        row = compute_metrics(dmnd, gt_pairs, gt_df,
                              {'method': 'diamond_fast', 'k': None, 'scheme': None, 'n_hash': None})
        results.append(row)
        log(f'  Diamond fast: {dmnd.height:,} hits after filters')
    else:
        log(f'WARNING: {DMND_FAST_TSV} not found — skipping (run --stage diamond first)')

    df = pl.DataFrame(results).sort(['method', 'scheme', 'k', 'n_hash'])
    df.write_csv(OUT_TSV, separator='\t')
    log(f'Results -> {OUT_TSV} ({len(df)} rows)')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['prep', 'sketch', 'diamond', 'eval'])
    args = parser.parse_args()

    if   args.stage == 'prep':    stage_prep()
    elif args.stage == 'sketch':  stage_sketch()
    elif args.stage == 'diamond': stage_diamond()
    elif args.stage == 'eval':    stage_eval()


if __name__ == '__main__':
    main()
