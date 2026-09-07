#!/usr/bin/env python3
"""
Search GTDB phylum-representative proteomes against AFDB and ESM Atlas diamond DBs.

Run from AT_DBs/. GTDB files are accessed from ../GTDB/r232/.

Stages (run in order):
  1. makedb   — build ESM diamond DB (skipped if .dmnd exists)
  2. concat   — concatenate rep_proteomes/*.faa.gz -> all_reps.faa (skipped if exists)
  3. search   — diamond blastp vs AFDB and ESM DBs (two parallel bsub jobs)

Usage:
  python search_gtdb_reps.py --stage makedb
  python search_gtdb_reps.py --stage concat
  python search_gtdb_reps.py --stage search
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

GTDB_DIR   = Path('../GTDB/r232')
WORK_DIR   = Path('.')

ESM_FA     = Path('esm_plddt60_non-afdb_reps.fasta')
ESM_DB     = Path('esm_plddt60_non-afdb_reps.dmnd')
AFDB_DB    = Path('afdb_uniref50_reps_db.dmnd')
ALL_REPS   = GTDB_DIR / 'all_reps.faa'
REP_DIR    = GTDB_DIR / 'rep_proteomes'

THREADS    = 128
OUTFMT     = '6 qseqid sseqid pident length qlen evalue'
EVALUE     = '1e-10'
PIDENT_MIN = 30  # post-filter only; diamond has no direct pident flag


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd, label=''):
    if label:
        log(label)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True)


def bsub(threads, queue, mem_gb, logbase, cmd):
    """Submit a job via bsub.py and return immediately."""
    full_cmd = (
        f'bsub.py --threads {threads} -q {queue} {mem_gb} '
        f'{logbase}.log "{cmd}"'
    )
    log(f'Submitting: {full_cmd}')
    subprocess.run(full_cmd, shell=True, check=True)


def stage_makedb():
    if ESM_DB.exists():
        log(f'Skipping makedb — {ESM_DB} already exists')
        return
    bsub(32, 'normal', 120, 'make_esmdb',
         f'diamond makedb --in {ESM_FA} --db esm_plddt60_non-afdb_reps --threads 32')


def stage_concat():
    if ALL_REPS.exists():
        log(f'Skipping concat — {ALL_REPS} already exists')
        return
    faa_files = sorted(REP_DIR.glob('*.faa.gz'))
    if not faa_files:
        sys.exit(f'ERROR: no .faa.gz files found in {REP_DIR}')
    log(f'Concatenating {len(faa_files)} .faa.gz files -> {ALL_REPS}')
    run(f'zcat {REP_DIR}/*.faa.gz > {ALL_REPS}')
    log('Done.')


def stage_search():
    if not ALL_REPS.exists():
        sys.exit(f'ERROR: {ALL_REPS} not found — run --stage concat first')

    hits_afdb = GTDB_DIR / 'gtdb_reps_vs_afdb.tsv'
    hits_esm  = GTDB_DIR / 'gtdb_reps_vs_esm.tsv'

    if hits_afdb.exists():
        log(f'Skipping AFDB search — {hits_afdb} already exists')
    else:
        bsub(THREADS, 'normal', 200, 'search_vs_afdb',
             f'diamond blastp -q {ALL_REPS} -d {AFDB_DB} -o {hits_afdb} '
             f'--outfmt {OUTFMT} --ultra-sensitive '
             f'--evalue {EVALUE} --min-score 0 --max-target-seqs 0 '
             f'--threads {THREADS}')

    if hits_esm.exists():
        log(f'Skipping ESM search — {hits_esm} already exists')
    else:
        bsub(THREADS, 'normal', 200, 'search_vs_esm',
             f'diamond blastp -q {ALL_REPS} -d {ESM_DB} -o {hits_esm} '
             f'--outfmt {OUTFMT} --ultra-sensitive '
             f'--evalue {EVALUE} --min-score 0 --max-target-seqs 0 '
             f'--threads {THREADS}')

    log(f'Note: filter pident >= {PIDENT_MIN}% from output TSVs after jobs complete (column 3)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['makedb', 'concat', 'search'],
                        help='Stage to run')
    args = parser.parse_args()

    if   args.stage == 'makedb':  stage_makedb()
    elif args.stage == 'concat':  stage_concat()
    elif args.stage == 'search':  stage_search()


if __name__ == '__main__':
    main()
