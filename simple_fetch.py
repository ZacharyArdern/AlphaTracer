#!/usr/bin/env python3
# simple_fetch.py  v0.1  —  AlphaTracer package
# Fetch PDB structures from AlphaFold DB and/or ESM Atlas given DIAMOND BLAST hits.
# Author: Zachary Ardern <z.ardern@gmail.com>
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
import asyncio
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp

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

# ── AFDB / ESM helpers ────────────────────────────────────────────────────────

AFDB_VERSION = 6
_ESM_DIR = os.environ.get('AT_ESM_DIR', str(Path.home() / 'Science/Data/ESMAtlas'))


def get_afdb_id(sseqid: str) -> str | None:
    """Extract bare AF-XXXX-F1 accession from a DIAMOND sseqid (strips version suffix)."""
    if sseqid.startswith('AF-'):
        acc = sseqid
    elif ':' in sseqid:
        acc = sseqid.split(':')[1]
        if not acc.startswith('AF-'):
            return None
    else:
        return None
    return re.sub(r'-model_v\d+$', '', acc)


def is_afdb_sseqid(sseqid: str) -> bool:
    return bool(get_afdb_id(sseqid))


def afdb_url(afdb_id: str) -> str:
    return f'https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v{AFDB_VERSION}.pdb'


def afdb_local_pdb(afdb_id: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'{afdb_id}-model_v{AFDB_VERSION}.pdb')


def esm_local_pdb(protein_hash: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'esm_{protein_hash}.pdb')


def is_valid_pdb(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            header = f.read(6).decode('ascii', errors='ignore')
        return header.strip()[:6] in ('HEADER', 'REMARK', 'ATOM  ', 'MODEL ')
    except Exception:
        return False


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


# ── AFDB fetch ────────────────────────────────────────────────────────────────

async def _fetch_afdb_pdbs_async(afdb_ids, pdb_dir: str) -> dict:
    sem = asyncio.Semaphore(64)
    connector = aiohttp.TCPConnector(limit=64)

    async def _fetch_one(session, aid):
        path = afdb_local_pdb(aid, pdb_dir)
        if os.path.exists(path) and is_valid_pdb(path):
            return aid, f'exists:{os.path.basename(path)}'
        url = afdb_url(aid)
        async with sem:
            try:
                async with session.get(url) as resp:
                    data = await resp.read()
                with open(path, 'wb') as f:
                    f.write(data)
                if is_valid_pdb(path):
                    return aid, f'downloaded:{os.path.basename(path)}'
                os.remove(path)
                return aid, f'failed:{aid}:server returned non-PDB content'
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)
                return aid, f'failed:{aid}:{e}'

    async with aiohttp.ClientSession(connector=connector) as session:
        return dict(await asyncio.gather(*[_fetch_one(session, aid) for aid in afdb_ids]))


def fetch_afdb_pdbs(afdb_ids, pdb_dir: str) -> dict:
    os.makedirs(pdb_dir, exist_ok=True)
    ids = list(afdb_ids)
    results = asyncio.run(_fetch_afdb_pdbs_async(ids, pdb_dir))
    retry = [aid for aid, r in results.items() if r.startswith('fail')]
    if retry:
        print(f'  Retrying {len(retry)} failed download(s)...')
        results.update(asyncio.run(_fetch_afdb_pdbs_async(retry, pdb_dir)))
    summary = Counter(r.split(':')[0] for r in results.values())
    for r in results.values():
        if r.startswith('fail'):
            print(f'  FAILED: {r}')
    print(f"  Downloaded: {summary['downloaded']}  "
          f"Already present: {summary['exists']}  "
          f"Failed: {summary.get('failed', 0) + summary.get('fail', 0)}")
    return results


# ── ESM Atlas fetch ───────────────────────────────────────────────────────────

def fetch_esm_structures(esm_rows, pdb_dir: str, n_workers: int = 8) -> None:
    esm_dir = os.path.abspath(_ESM_DIR)
    if esm_dir not in sys.path:
        sys.path.insert(0, esm_dir)
    try:
        import esm_query as _esm
    except ImportError:
        print(f'  [WARN] Cannot import esm_query from {esm_dir} — ESM Atlas hits skipped')
        return

    hits, need_lookup, seen = [], [], set()
    for row in esm_rows:
        ph  = row.get('protein_hash') or ''
        fid = row.get('fragment_id', -1)
        fr  = row.get('frag_row', -1)
        if not ph or ph in seen:
            continue
        seen.add(ph)
        if os.path.exists(esm_local_pdb(ph, pdb_dir)):
            continue
        if fid is not None and fid >= 0 and fr is not None and fr >= 0:
            hits.append({'fragment_id': fid, 'frag_row': fr, 'protein_hash': ph})
        else:
            need_lookup.append(ph)

    if need_lookup:
        print(f'  Resolving {len(need_lookup)} ESM protein_hash(es) via local index...',
              flush=True)
        try:
            idx = _esm.lookup_hashes(need_lookup)
            lookup_map = {r['protein_hash']: (r['fragment_id'], r['frag_row'])
                          for r in [dict(zip(idx.schema.names, row))
                                    for row in zip(*[idx.column(c).to_pylist()
                                                     for c in idx.schema.names])]}
            for ph in need_lookup:
                coords = lookup_map.get(ph)
                if coords and coords[0] >= 0:
                    hits.append({'fragment_id': int(coords[0]),
                                 'frag_row': int(coords[1]),
                                 'protein_hash': ph})
                else:
                    print(f'  [WARN] No index entry for ESM protein_hash {ph} — skipping')
        except Exception as e:
            print(f'  [WARN] ESM lookup_hashes failed: {e}')

    if not hits:
        return

    print(f'  Fetching {len(hits)} ESM Atlas PDB(s) from S3...', flush=True)
    t0 = time.time()
    try:
        pdb_table = _esm.query_from_hits(hits,
                                         columns=['protein_hash', 'structure_blob'],
                                         n_workers=n_workers)
    except Exception as e:
        print(f'  [WARN] ESM Atlas fetch failed: {type(e).__name__}: {e}')
        return

    n_pdb = 0
    if pdb_table is not None:
        for batch in pdb_table.to_batches():
            for ph, blob in zip(batch['protein_hash'].to_pylist(),
                                batch['structure_blob'].to_pylist()):
                if blob is None:
                    continue
                try:
                    with open(esm_local_pdb(ph, pdb_dir), 'w') as f:
                        f.write(_esm.blob_to_pdb(blob))
                    n_pdb += 1
                except Exception as e:
                    print(f'  [WARN] ESM decode failed for {ph}: {e}')

    print(f'  Fetched {n_pdb}/{len(hits)} ESM structures in {time.time()-t0:.1f}s',
          flush=True)


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
