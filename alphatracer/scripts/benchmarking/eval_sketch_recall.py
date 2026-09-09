#!/usr/bin/env python3
"""
Evaluate recall and precision of sketch search vs diamond ground truth.

Run from AT_DBs/. Loads afdb_hits_merged.tsv as ground truth, evaluates
each sketch_hits/*.tsv.gz file one at a time, and writes a summary TSV + plots.

Recall per pident bin: fraction of diamond hits in that bin found by sketch.
Precision overall:     fraction of sketch hits that have any diamond match.

Usage:
  python eval_sketch_recall.py
  python eval_sketch_recall.py --diamond afdb_hits_merged.tsv --hits-dir sketch_hits/
"""
import argparse
import re
import time
from pathlib import Path

import polars as pl

DIAMOND_TSV = Path('afdb_hits_merged.tsv')
DIAMOND_PQ  = Path('afdb_hits_filtered.pq')
HITS_DIR    = Path('sketch_hits')
OUT_TSV     = Path('sketch_eval_results.tsv')
OUT_PLOT    = Path('sketch_eval_plots.png')

EVALUE_MAX  = 1e-10
MIN_ALN_LEN = 40
PIDENT_BINS = [(30, 40), (40, 50), (50, 60), (60, 101)]
BIN_LABELS  = ['30-40%', '40-50%', '50-60%', '60%+']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def prep_diamond(tsv_path: Path, pq_path: Path) -> None:
    """Stream diamond TSV, filter, and write compact parquet. Skipped if parquet exists."""
    if pq_path.exists():
        log(f'Skipping diamond prep — {pq_path} already exists')
        return
    log(f'Filtering {tsv_path} -> {pq_path} (streaming, may take a while)...')
    t0 = time.time()
    (
        pl.scan_csv(tsv_path, separator='\t', has_header=False,
                    new_columns=['query', 'target', 'pident', 'length', 'qlen', 'evalue'],
                    schema_overrides={'pident': pl.Float32, 'length': pl.Int32,
                                      'evalue': pl.Float64})
        .filter(pl.col('evalue') <= EVALUE_MAX)
        .filter(pl.col('pident') >= 30)
        .filter(pl.col('length') >= MIN_ALN_LEN)
        .select(['query', 'target', 'pident'])
        .with_columns(
            pl.when(pl.col('pident').is_between(30, 40, closed='left')).then(pl.lit('30-40%'))
            .when(pl.col('pident').is_between(40, 50, closed='left')).then(pl.lit('40-50%'))
            .when(pl.col('pident').is_between(50, 60, closed='left')).then(pl.lit('50-60%'))
            .otherwise(pl.lit('60%+'))
            .alias('pident_bin')
        )
        .sink_parquet(pq_path, compression='zstd')
    )
    log(f'Done in {time.time()-t0:.1f}s -> {pq_path}')


def log_diamond_counts(pq_path: Path) -> None:
    log(f'Diamond ground truth counts from {pq_path}:')
    counts = (
        pl.scan_parquet(pq_path)
        .group_by('pident_bin')
        .agg(pl.len().alias('n'))
        .collect()
        .sort('pident_bin')
    )
    total = 0
    for row in counts.to_dicts():
        log(f'  {row["pident_bin"]}: {row["n"]:,}')
        total += row['n']
    log(f'  Total: {total:,}')


def parse_job_name(stem: str) -> tuple | None:
    m = re.search(r'k(\d+)_(murphy\w+|dayhoff\w+)_n(\d+)', stem)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def eval_one(path: Path, diamond_pq: Path) -> dict | None:
    params = parse_job_name(path.stem)
    if params is None:
        return None
    k, scheme, n_hash = params

    try:
        sketch = pl.read_csv(path, separator='\t', has_header=True,
                             schema_overrides={'n_shared': pl.UInt32,
                                              'containment': pl.Float32})
    except Exception as e:
        log(f'  WARNING: could not read {path.name}: {e}')
        return None

    n_sketch = sketch.height
    if n_sketch == 0:
        return None

    sketch_pairs = sketch.select(['query_id', 'target_id'])

    # ── Recall: lazy join diamond against sketch, aggregate only ─────────────
    recall_df = (
        pl.scan_parquet(diamond_pq)
        .join(
            sketch_pairs.lazy().with_columns(pl.lit(True).alias('found')),
            left_on=['query', 'target'],
            right_on=['query_id', 'target_id'],
            how='left'
        )
        .with_columns(pl.col('found').fill_null(False))
        .group_by('pident_bin')
        .agg([pl.len().alias('n_gt'), pl.col('found').sum().alias('n_tp')])
        .with_columns((pl.col('n_tp') / pl.col('n_gt')).alias('recall'))
        .collect(streaming=True)
    )

    # ── Precision: lazy semi-join sketch against diamond ─────────────────────
    n_tp_total = (
        sketch_pairs.lazy()
        .join(pl.scan_parquet(diamond_pq).select(['query', 'target']),
              left_on=['query_id', 'target_id'],
              right_on=['query', 'target'],
              how='semi')
        .collect(streaming=True)
        .height
    )
    precision_overall = n_tp_total / n_sketch

    # ── Collect results ───────────────────────────────────────────────────────
    row = {'k': k, 'scheme': scheme, 'n_hash': n_hash,
           'n_sketch': n_sketch, 'precision_overall': round(precision_overall, 4)}

    recall_map = {r['pident_bin']: r for r in recall_df.to_dicts()}
    for label in BIN_LABELS:
        r = recall_map.get(label, {})
        row[f'recall_{label}']  = round(r.get('recall', 0.0), 4)
        row[f'n_gt_{label}']    = r.get('n_gt', 0)
        row[f'n_tp_{label}']    = r.get('n_tp', 0)

    del sketch
    return row


def make_plots(df: pl.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    schemes  = sorted(df['scheme'].unique().to_list())
    n_hashes = sorted(df['n_hash'].unique().to_list())
    colors   = plt.cm.tab10.colors
    styles   = ['-', '--', ':', '-.']

    n_rows = len(BIN_LABELS) + 1  # extra row for precision
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)

    for row_i, label in enumerate(BIN_LABELS):
        ax_rec  = axes[row_i][0]
        ax_prec = axes[row_i][1]
        for si, scheme in enumerate(schemes):
            for ni, n_hash in enumerate(n_hashes):
                sub = df.filter(
                    (pl.col('scheme') == scheme) & (pl.col('n_hash') == n_hash)
                ).sort('k')
                if sub.is_empty():
                    continue
                ks      = sub['k'].to_list()
                recalls = sub[f'recall_{label}'].to_list()
                precs   = sub['precision_overall'].to_list()
                lbl = f'{scheme} n={n_hash}'
                c, ls = colors[si % len(colors)], styles[ni % len(styles)]
                ax_rec.plot(ks,  recalls, linestyle=ls, color=c, label=lbl, marker='o', ms=3)
                ax_prec.plot(ks, precs,   linestyle=ls, color=c, label=lbl, marker='o', ms=3)

        ax_rec.set_title(f'Recall — pident {label}')
        ax_rec.set_xlabel('k'); ax_rec.set_ylabel('Recall')
        ax_rec.set_ylim(0, 1.05); ax_rec.set_xticks(range(6, 16)); ax_rec.grid(True, alpha=0.3)
        ax_prec.set_title(f'Precision (overall) at pident {label}')
        ax_prec.set_xlabel('k'); ax_prec.set_ylabel('Precision')
        ax_prec.set_ylim(0, 1.05); ax_prec.set_xticks(range(6, 16)); ax_prec.grid(True, alpha=0.3)

    # Bottom row: precision only
    ax_p = axes[len(BIN_LABELS)][0]
    for si, scheme in enumerate(schemes):
        for ni, n_hash in enumerate(n_hashes):
            sub = df.filter(
                (pl.col('scheme') == scheme) & (pl.col('n_hash') == n_hash)
            ).sort('k')
            if sub.is_empty():
                continue
            ax_p.plot(sub['k'].to_list(), sub['precision_overall'].to_list(),
                      linestyle=styles[ni % len(styles)], color=colors[si % len(colors)],
                      label=f'{scheme} n={n_hash}', marker='o', ms=3)
    ax_p.set_title('Precision overall (any diamond match)')
    ax_p.set_xlabel('k'); ax_p.set_ylabel('Precision')
    ax_p.set_ylim(0, 1.05); ax_p.set_xticks(range(6, 16)); ax_p.grid(True, alpha=0.3)
    axes[len(BIN_LABELS)][1].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    log(f'Plot saved -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--diamond',  type=Path, default=DIAMOND_TSV)
    parser.add_argument('--hits-dir', type=Path, default=HITS_DIR)
    parser.add_argument('--out',      type=Path, default=OUT_TSV)
    parser.add_argument('--plot',     type=Path, default=OUT_PLOT)
    parser.add_argument('--no-plot',  action='store_true')
    args = parser.parse_args()

    if not args.diamond.exists() and not DIAMOND_PQ.exists():
        raise SystemExit(f'ERROR: {args.diamond} not found')

    if args.diamond.exists():
        prep_diamond(args.diamond, DIAMOND_PQ)
    log_diamond_counts(DIAMOND_PQ)

    hits_files = sorted(args.hits_dir.glob('*.tsv.gz'))
    if not hits_files:
        raise SystemExit(f'ERROR: no .tsv.gz files in {args.hits_dir}')
    log(f'Evaluating {len(hits_files)} sketch files...')

    results = []
    for i, path in enumerate(hits_files):
        row = eval_one(path, DIAMOND_PQ)
        if row:
            results.append(row)
        if (i + 1) % 20 == 0:
            log(f'  {i+1}/{len(hits_files)} done')

    df = pl.DataFrame(results).sort(['scheme', 'k', 'n_hash'])
    df.write_csv(args.out, separator='\t')
    log(f'Results -> {args.out} ({len(df)} rows)')

    if not args.no_plot:
        try:
            make_plots(df, args.plot)
        except ImportError:
            log('matplotlib not available — skipping plots (pip install matplotlib)')


if __name__ == '__main__':
    main()
