#!/usr/bin/env python3
# simple_fetch.py  v0.1  —  AlphaTracer package
# Fetch PDB structures from AlphaFold DB and/or ESM Atlas given DIAMOND BLAST hits.
# Author: Zachary Ardern <z.ardern@gmail.com>
#
# Requires AlphaTracer to be installed:
#   pip install -e /path/to/AlphaTracer
#
# ── Mode 1: run DIAMOND then fetch (default) ──────────────────────────────────
#   simple_fetch.py -q query.fasta -d afdb.dmnd esm.dmnd -o hits \
#       --id 50 --qcov 80 --targets 1 --outdir pdb_hits/
#
#   DIAMOND command constructed internally (per database):
#     diamond blastp -q query.fasta -d afdb.dmnd -o hits_1.tsv \
#         --outfmt 6 qseqid sseqid pident qcovhsp bitscore \
#         --max-target-seqs <targets*5> --id <id> --query-cover <qcov> \
#         --threads <threads>
#
# ── Mode 2: fetch from existing hits files ────────────────────────────────────
#   simple_fetch.py --afdb_hits hits_1.tsv --esm_hits hits_2.tsv \
#       --fmt qseqid sseqid pident qcovhsp bitscore \
#       --pident 30 --qcov 50 --targets 1 --outdir pdb_hits/

import argparse
import os
import subprocess
import sys
from collections import defaultdict

try:
    from alphatracer.utils.afdb_fetch import get_afdb_id, fetch_afdb_pdbs
    from alphatracer.utils.esm_atlas_fetch import fetch_esm_structures
except ImportError:
    import os as _os
    _search = [
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'alphatracer', 'utils'),
        _os.getcwd(),
    ]
    for _p in _search:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from afdb_fetch import get_afdb_id, fetch_afdb_pdbs
        from esm_atlas_fetch import fetch_esm_structures
    except ImportError:
        sys.exit(
            'Error: could not import afdb_fetch / esm_atlas_fetch.\n'
            'Options:\n'
            '  1. Install AlphaTracer:  pip install -e /path/to/AlphaTracer\n'
            '  2. Run from the AlphaTracer repo directory\n'
            '  3. Copy afdb_fetch.py and esm_atlas_fetch.py to the current directory'
        )

VERSION = "0.1"

BANNER = (
    f"simple_fetch.py  v{VERSION}  —  AlphaTracer package\n"
    "Fetch PDB structures from AlphaFold DB and/or ESM Atlas.\n"
    "Author: Zachary Ardern <z.ardern@gmail.com>"
)

# Fixed outfmt used when simple_fetch runs DIAMOND itself
SEARCH_FMT = ['qseqid', 'sseqid', 'pident', 'qcovhsp', 'bitscore']

# Default DIAMOND --outfmt 6 columns (fetch-only mode)
DEFAULT_FMT = (
    "qseqid sseqid pident length mismatch gapopen "
    "qstart qend sstart send evalue bitscore"
).split()


def is_afdb_sseqid(sseqid: str) -> bool:
    return bool(get_afdb_id(sseqid))


def infer_db_tag(hits_path: str) -> str:
    """Peek at the first data line of a hits file and infer 'afdb' or 'esm'."""
    try:
        with open(hits_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                sseqid = line.split('\t')[1] if '\t' in line else ''
                return 'afdb' if is_afdb_sseqid(sseqid) else 'esm'
    except Exception:
        pass
    return 'esm'


# ── DIAMOND search ────────────────────────────────────────────────────────────

def run_diamond(query: str, db: str, output: str,
                pident: float, qcov: float, targets: int, threads: int) -> None:
    max_targets = max(targets * 5, 25)
    cmd = [
        'diamond', 'blastp',
        '-q', query,
        '-d', db,
        '-o', output,
        '--outfmt', '6', *SEARCH_FMT,
        '--max-target-seqs', str(max_targets),
        '--id', str(pident),
        '--query-cover', str(qcov),
        '--threads', str(threads),
    ]
    print(f'  Running: {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, check=True)


# ── Hit parsing and selection ─────────────────────────────────────────────────

def parse_hits(path, fmt_fields, pident_min, qcov_min, db_tag):
    hits = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != len(fmt_fields):
                print(f'  [WARN] {path}:{lineno}: expected {len(fmt_fields)} fields, '
                      f'got {len(parts)} — skipping')
                continue
            row = dict(zip(fmt_fields, parts))
            row['_db'] = db_tag
            for field in ('pident', 'qcovhsp', 'bitscore', 'evalue', 'score'):
                if field in row:
                    try:
                        row[field] = float(row[field])
                    except ValueError:
                        pass
            pident = row.get('pident')
            qcov   = row.get('qcovhsp')
            if pident_min > 0 and isinstance(pident, float) and pident < pident_min:
                continue
            if qcov_min > 0 and isinstance(qcov, float) and qcov < qcov_min:
                continue
            hits.append(row)
    return hits


def select_top_hits(all_hits, targets, sort_field):
    by_query = defaultdict(list)
    for h in all_hits:
        by_query[h['qseqid']].append(h)
    selected = []
    for qid, qhits in sorted(by_query.items()):
        if sort_field:
            qhits.sort(key=lambda h: h.get(sort_field, 0.0)
                       if isinstance(h.get(sort_field), float) else 0.0,
                       reverse=True)
        selected.extend(qhits[:targets])
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    print()

    parser = argparse.ArgumentParser(
        description='Run DIAMOND and fetch PDB files from AlphaFold DB and/or ESM Atlas.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Search mode (default)
    search_grp = parser.add_argument_group('Search mode (runs DIAMOND then fetches)')
    search_grp.add_argument('--search', action='store_true',
                            help='Explicit flag for search mode (default if -q/-d/-o given)')
    search_grp.add_argument('-q', metavar='FASTA',
                            help='Query FASTA file')
    search_grp.add_argument('-d', nargs='+', metavar='DB',
                            help='DIAMOND database(s) (.dmnd), 1 or 2')
    search_grp.add_argument('-o', metavar='BASE',
                            help='Output base name for DIAMOND hits '
                                 '(produces <base>_1.tsv [and <base>_2.tsv])')
    search_grp.add_argument('--id', dest='id', type=float, default=50.0,
                            help='DIAMOND --id: minimum identity %% (default: 50)')
    search_grp.add_argument('--qcov', type=float, default=80.0,
                            help='DIAMOND --query-cover: minimum query coverage %% (default: 80)')
    search_grp.add_argument('--threads', type=int,
                            default=os.cpu_count() or 4,
                            help='Threads for DIAMOND (default: all CPUs)')

    # Fetch-only mode
    fetch_grp = parser.add_argument_group('Fetch-only mode (use existing hits files)')
    fetch_grp.add_argument('--afdb_hits', metavar='FILE',
                           help='Pre-computed DIAMOND hits file against AFDB')
    fetch_grp.add_argument('--esm_hits', metavar='FILE',
                           help='Pre-computed DIAMOND hits file against ESM Atlas')
    fetch_grp.add_argument('--fmt', nargs='+', default=DEFAULT_FMT, metavar='FIELD',
                           help='Column names for hits files (default: standard 12-column fmt 6)')
    fetch_grp.add_argument('--pident', type=float, default=0.0,
                           help='Post-filter: minimum percent identity (default: 0)')

    # Shared
    parser.add_argument('--targets', type=int, default=1,
                        help='Best hits to keep per query, ranked by bitscore (default: 1)')
    parser.add_argument('--outdir', default='pdb_hits',
                        help='Output directory for PDB files (default: pdb_hits)')

    args = parser.parse_args()

    # ── Determine mode ────────────────────────────────────────────────────────

    search_mode = args.search or bool(args.q or args.d or args.o)
    fetch_only  = bool(args.afdb_hits or args.esm_hits)

    if not search_mode and not fetch_only:
        parser.error(
            'Provide either:\n'
            '  Search mode:     -q FASTA -d DB [DB] -o BASE\n'
            '  Fetch-only mode: --afdb_hits FILE [--esm_hits FILE]'
        )

    if search_mode and fetch_only:
        parser.error('Cannot mix search mode (-q/-d/-o) with fetch-only mode '
                     '(--afdb_hits/--esm_hits)')

    # ── Search mode ───────────────────────────────────────────────────────────

    hits_files = []   # list of (path, db_tag)

    if search_mode:
        for flag, name in [('-q', args.q), ('-d', args.d), ('-o', args.o)]:
            if not name:
                parser.error(f'{flag} is required in search mode')
        if len(args.d) > 2:
            parser.error('-d accepts at most 2 databases')

        print(f'Search: {len(args.d)} database(s), '
              f'--id {args.id}  --qcov {args.qcov}  --targets {args.targets}  '
              f'--threads {args.threads}\n')

        for i, db in enumerate(args.d, 1):
            out_path = f'{args.o}_{i}.tsv'
            print(f'[{i}/{len(args.d)}] Searching against {db} → {out_path}')
            run_diamond(args.q, db, out_path, args.id, args.qcov, args.targets, args.threads)
            db_tag = infer_db_tag(out_path)
            print(f'       Detected database type: {db_tag.upper()}')
            hits_files.append((out_path, db_tag))

        fmt = SEARCH_FMT
        pident_filter = 0.0   # DIAMOND already applied --id
        qcov_filter   = 0.0   # DIAMOND already applied --query-cover

    # ── Fetch-only mode ───────────────────────────────────────────────────────

    else:
        fmt = args.fmt
        if 'qseqid' not in fmt or 'sseqid' not in fmt:
            parser.error('--fmt must include qseqid and sseqid')
        if 'qcovhsp' not in fmt and args.qcov > 0:
            print('[WARN] --qcov filter requested but qcovhsp not in --fmt; filter skipped')
        if 'bitscore' not in fmt:
            print('[WARN] bitscore not in --fmt — bitscore is recommended for ranking hits '
                  'across databases, as it accounts for alignment length unlike pident alone')

        pident_filter = args.pident
        qcov_filter   = args.qcov

        if args.afdb_hits:
            hits_files.append((args.afdb_hits, 'afdb'))
        if args.esm_hits:
            hits_files.append((args.esm_hits, 'esm'))

    # ── Sort field ────────────────────────────────────────────────────────────

    if 'bitscore' in fmt:
        sort_field = 'bitscore'
    elif 'pident' in fmt:
        sort_field = 'pident'
        if not search_mode:
            print('[WARN] Falling back to pident for sorting')
    else:
        sort_field = None
        print('[WARN] Neither bitscore nor pident in fmt; hits will not be sorted')

    # ── Parse and filter hits ─────────────────────────────────────────────────

    print()
    all_hits = []
    for path, db_tag in hits_files:
        hits = parse_hits(path, fmt, pident_filter, qcov_filter, db_tag)
        print(f'{db_tag.upper():5s}: {len(hits)} hits  ({path})')
        all_hits.extend(hits)

    if not all_hits:
        print('No hits pass filters. Exiting.')
        sys.exit(0)

    selected = select_top_hits(all_hits, args.targets, sort_field)
    afdb_sel = [h for h in selected if h['_db'] == 'afdb']
    esm_sel  = [h for h in selected if h['_db'] == 'esm']

    n_queries = len({h['qseqid'] for h in selected})
    print(f'\nSelected {len(selected)} hits for {n_queries} '
          f'{"query" if n_queries == 1 else "queries"} '
          f'({len(afdb_sel)} AFDB, {len(esm_sel)} ESM)')

    os.makedirs(args.outdir, exist_ok=True)

    # ── Fetch AFDB ────────────────────────────────────────────────────────────

    if afdb_sel:
        afdb_ids = []
        for h in afdb_sel:
            aid = get_afdb_id(h['sseqid'])
            if aid:
                afdb_ids.append(aid)
            else:
                print(f'  [WARN] Cannot parse AFDB accession from sseqid: {h["sseqid"]}')
        unique_ids = list(dict.fromkeys(afdb_ids))
        if unique_ids:
            print(f'\nFetching {len(unique_ids)} AFDB PDB file(s) → {args.outdir}/')
            fetch_afdb_pdbs(unique_ids, args.outdir)

    # ── Fetch ESM Atlas ───────────────────────────────────────────────────────

    if esm_sel:
        esm_rows = [{'protein_hash': h['sseqid']} for h in esm_sel]
        seen, unique_rows = set(), []
        for r in esm_rows:
            if r['protein_hash'] not in seen:
                seen.add(r['protein_hash'])
                unique_rows.append(r)
        print(f'\nFetching {len(unique_rows)} ESM Atlas PDB file(s) → {args.outdir}/')
        fetch_esm_structures(unique_rows, pdb_dir=args.outdir)

    print(f'\nDone.  PDB files written to: {os.path.abspath(args.outdir)}/')


if __name__ == '__main__':
    main()
