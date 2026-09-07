#!/usr/bin/env python3
"""
Search GTDB phylum-representative proteomes against AFDB and ESM Atlas diamond DBs.

Run from AT_DBs/. GTDB files are accessed from ../GTDB/r232/.

Stages (run in order):
  1. concat   — concatenate rep_proteomes/*.faa.gz -> all_reps.faa (skipped if exists)
  2. split    — split AFDB (10 chunks) and ESM (20 chunks) fastas via seqkit
  3. makedb   — submit bsub makedb jobs for each chunk (30 jobs total)
  4. search   — submit bsub search jobs for each chunk (30 jobs total)
  5. merge    — cat chunk hit TSVs into afdb_hits_merged.tsv and esm_hits_merged.tsv

Usage:
  python search_reps_dmnd.py --stage concat
  python search_reps_dmnd.py --stage split
  python search_reps_dmnd.py --stage makedb
  python search_reps_dmnd.py --stage search
  python search_reps_dmnd.py --stage merge
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

GTDB_DIR       = Path('../GTDB/r232')
WORK_DIR       = Path('.')

AFDB_FA        = Path('AFDBv6_uniref50_reps.fasta')
ESM_FA         = Path('esm_plddt60_non-afdb_reps.fasta')
AFDB_CHUNK_DIR = Path('afdb_reps_chunks')
ESM_CHUNK_DIR  = Path('esm_reps_chunks')
AFDB_HITS_DIR  = Path('afdb_hits')
ESM_HITS_DIR   = Path('esm_hits')

ALL_REPS       = GTDB_DIR / 'all_reps.faa'
REP_DIR        = GTDB_DIR / 'rep_proteomes'

N_AFDB_CHUNKS  = 10
N_ESM_CHUNKS   = 20
SEARCH_THREADS = 64
MAKEDB_THREADS = 8
OUTFMT         = '6 qseqid sseqid pident length qlen evalue'
EVALUE         = '1e-10'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd, label=''):
    if label:
        log(label)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True)


def bsub(threads, queue, mem_gb, logbase, cmd):
    full_cmd = (
        f'bsub.py --threads {threads} -q {queue} {mem_gb} '
        f'{logbase}.log "{cmd}"'
    )
    log(f'Submitting: {full_cmd}')
    subprocess.run(full_cmd, shell=True, check=True)


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


def stage_split():
    AFDB_CHUNK_DIR.mkdir(exist_ok=True)
    ESM_CHUNK_DIR.mkdir(exist_ok=True)

    afdb_chunks = sorted(AFDB_CHUNK_DIR.glob('*.fasta'))
    if len(afdb_chunks) >= N_AFDB_CHUNKS:
        log(f'Skipping AFDB split — {len(afdb_chunks)} chunks already in {AFDB_CHUNK_DIR}')
    else:
        if not AFDB_FA.exists():
            sys.exit(f'ERROR: {AFDB_FA} not found')
        run(f'seqkit split2 -p {N_AFDB_CHUNKS} {AFDB_FA} -O {AFDB_CHUNK_DIR} --quiet',
            f'Splitting AFDB into {N_AFDB_CHUNKS} chunks...')

    esm_chunks = sorted(ESM_CHUNK_DIR.glob('*.fasta'))
    if len(esm_chunks) >= N_ESM_CHUNKS:
        log(f'Skipping ESM split — {len(esm_chunks)} chunks already in {ESM_CHUNK_DIR}')
    else:
        if not ESM_FA.exists():
            sys.exit(f'ERROR: {ESM_FA} not found')
        run(f'seqkit split2 -p {N_ESM_CHUNKS} {ESM_FA} -O {ESM_CHUNK_DIR} --quiet',
            f'Splitting ESM into {N_ESM_CHUNKS} chunks...')


def stage_makedb():
    for chunk_dir, label in [(AFDB_CHUNK_DIR, 'afdb'), (ESM_CHUNK_DIR, 'esm')]:
        chunks = sorted(chunk_dir.glob('*.fasta'))
        if not chunks:
            sys.exit(f'ERROR: no chunks in {chunk_dir} — run --stage split first')
        for chunk in chunks:
            db = chunk.with_suffix('.dmnd')
            if db.exists():
                log(f'Skipping makedb — {db} exists')
                continue
            bsub(MAKEDB_THREADS, 'normal', 32,
                 f'makedb_{label}_{chunk.stem}',
                 f'diamond makedb --in {chunk} --db {chunk.with_suffix("")} --threads {MAKEDB_THREADS}')


def stage_search():
    if not ALL_REPS.exists():
        sys.exit(f'ERROR: {ALL_REPS} not found — run --stage concat first')

    AFDB_HITS_DIR.mkdir(exist_ok=True)
    ESM_HITS_DIR.mkdir(exist_ok=True)

    for chunk_dir, hits_dir, label in [
        (AFDB_CHUNK_DIR, AFDB_HITS_DIR, 'afdb'),
        (ESM_CHUNK_DIR,  ESM_HITS_DIR,  'esm'),
    ]:
        chunks = sorted(chunk_dir.glob('*.fasta'))
        if not chunks:
            sys.exit(f'ERROR: no chunks in {chunk_dir} — run --stage split first')
        for chunk in chunks:
            db   = chunk.with_suffix('.dmnd')
            hits = hits_dir / f'{chunk.stem}.tsv'
            if not db.exists():
                log(f'WARNING: {db} not found — skipping (run --stage makedb first)')
                continue
            if hits.exists():
                log(f'Skipping search — {hits} exists')
                continue
            bsub(SEARCH_THREADS, 'normal', 64,
                 f'search_vs_{label}_{chunk.stem}',
                 f'diamond blastp -q {ALL_REPS} -d {db} -o {hits} '
                 f'--outfmt {OUTFMT} --ultra-sensitive '
                 f'--evalue {EVALUE} --min-score 0 --max-target-seqs 0 '
                 f'--threads {SEARCH_THREADS}')


def stage_merge():
    for hits_dir, out_name, label in [
        (AFDB_HITS_DIR, 'afdb_hits_merged.tsv', 'AFDB'),
        (ESM_HITS_DIR,  'esm_hits_merged.tsv',  'ESM'),
    ]:
        out = WORK_DIR / out_name
        if out.exists():
            log(f'Skipping {label} merge — {out} exists')
            continue
        tsvs = sorted(hits_dir.glob('*.tsv'))
        if not tsvs:
            log(f'WARNING: no TSVs found in {hits_dir} — skipping {label} merge')
            continue
        run(f'cat {" ".join(str(t) for t in tsvs)} > {out}',
            f'Merging {len(tsvs)} {label} hit files -> {out}')
        n = sum(1 for _ in open(out))
        log(f'  {n:,} hits in {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True,
                        choices=['concat', 'split', 'makedb', 'search', 'merge'],
                        help='Stage to run')
    args = parser.parse_args()

    if   args.stage == 'concat': stage_concat()
    elif args.stage == 'split':  stage_split()
    elif args.stage == 'makedb': stage_makedb()
    elif args.stage == 'search': stage_search()
    elif args.stage == 'merge':  stage_merge()


if __name__ == '__main__':
    main()
