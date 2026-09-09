#!/usr/bin/env python3
"""
Evaluate recall and precision of sketch search vs diamond ground truth.

Run from AT_DBs/. Loads afdb_hits_merged.tsv as ground truth, evaluates
each sketch_hits/*.tsv.gz file, and writes a summary TSV + plots.

Usage:
  python eval_sketch_recall.py
  python eval_sketch_recall.py --diamond afdb_hits_merged.tsv --hits-dir sketch_hits/
"""
import argparse
import csv
import gzip
import re
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import polars as pl

DIAMOND_TSV = Path('afdb_hits_merged.tsv')
HITS_DIR    = Path('sketch_hits')
OUT_TSV     = Path('sketch_eval_results.tsv')
OUT_PLOT    = Path('sketch_eval_plots.png')

PIDENT_BINS = [(30, 40), (40, 50), (50, 60), (60, 101)]
BIN_LABELS  = ['30-40%', '40-50%', '50-60%', '60%+']
EVALUE_MAX  = 1e-10


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_diamond(path: Path) -> dict[str, set]:
    """Load diamond hits into dict: bin_label -> set of (query, target) pairs."""
    log(f'Loading diamond ground truth from {path}...')
    gt = {label: set() for label in BIN_LABELS}
    n = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 6:
                continue
            query, target, pident, _, _, evalue = parts[:6]
            try:
                pident = float(pident)
                evalue = float(evalue)
            except ValueError:
                continue
            if evalue > EVALUE_MAX:
                continue
            for (lo, hi), label in zip(PIDENT_BINS, BIN_LABELS):
                if lo <= pident < hi:
                    gt[label].add((query, target))
                    n += 1
                    break
    for label in BIN_LABELS:
        log(f'  {label}: {len(gt[label]):,} ground-truth pairs')
    log(f'  Total: {n:,} pairs across all bins')
    return gt


def eval_sketch_file(path: Path, gt: dict[str, set]) -> dict | None:
    """Compute recall and precision per pident bin for one sketch hits file."""
    m = re.search(r'k(\d+)_(murphy\d+|dayhoff\w+)_n(\d+)', path.stem)
    if not m:
        return None
    k, scheme, n_hash = int(m.group(1)), m.group(2), int(m.group(3))

    sketch_pairs = set()
    try:
        with gzip.open(path, 'rt') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                sketch_pairs.add((row['query_id'], row['target_id']))
    except Exception as e:
        log(f'  WARNING: could not read {path}: {e}')
        return None

    row_out = {'k': k, 'scheme': scheme, 'n_hash': n_hash,
               'n_sketch_total': len(sketch_pairs)}
    for label, (lo, hi) in zip(BIN_LABELS, PIDENT_BINS):
        gt_bin = gt[label]
        tp = len(sketch_pairs & gt_bin)
        fn = len(gt_bin) - tp
        fp = len(sketch_pairs) - tp  # all sketch hits not in this gt bin
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        row_out[f'recall_{label}']    = round(recall, 4)
        row_out[f'precision_{label}'] = round(precision, 4)
        row_out[f'n_gt_{label}']      = len(gt_bin)
        row_out[f'n_tp_{label}']      = tp
    return row_out


def make_plots(df: pl.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    schemes = df['scheme'].unique().sort().to_list()
    n_hashes = sorted(df['n_hash'].unique().to_list())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(len(BIN_LABELS), 2,
                             figsize=(14, 4 * len(BIN_LABELS)),
                             squeeze=False)

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
                precs   = sub[f'precision_{label}'].to_list()
                lbl = f'{scheme} n={n_hash}'
                ls  = ['-', '--', ':', '-.'][ni]
                c   = colors[si % len(colors)]
                ax_rec.plot(ks, recalls, linestyle=ls, color=c, label=lbl, marker='o', ms=3)
                ax_prec.plot(ks, precs,  linestyle=ls, color=c, label=lbl, marker='o', ms=3)

        ax_rec.set_title(f'Recall — pident {label}')
        ax_rec.set_xlabel('k'); ax_rec.set_ylabel('Recall')
        ax_rec.set_ylim(0, 1.05); ax_rec.set_xticks(range(6, 16))
        ax_rec.grid(True, alpha=0.3)

        ax_prec.set_title(f'Precision — pident {label}')
        ax_prec.set_xlabel('k'); ax_prec.set_ylabel('Precision')
        ax_prec.set_ylim(0, 1.05); ax_prec.set_xticks(range(6, 16))
        ax_prec.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
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

    if not args.diamond.exists():
        raise SystemExit(f'ERROR: {args.diamond} not found')

    gt = load_diamond(args.diamond)

    hits_files = sorted(args.hits_dir.glob('*.tsv.gz'))
    if not hits_files:
        raise SystemExit(f'ERROR: no .tsv.gz files found in {args.hits_dir}')
    log(f'Evaluating {len(hits_files)} sketch result files...')

    results = []
    for i, path in enumerate(hits_files):
        row = eval_sketch_file(path, gt)
        if row:
            results.append(row)
        if (i + 1) % 20 == 0:
            log(f'  {i+1}/{len(hits_files)} done')

    df = pl.DataFrame(results).sort(['scheme', 'k', 'n_hash'])
    df.write_csv(args.out, separator='\t')
    log(f'Results written -> {args.out} ({len(df)} rows)')

    if not args.no_plot:
        try:
            make_plots(df, args.plot)
        except ImportError:
            log('matplotlib not available — skipping plots')


if __name__ == '__main__':
    main()
