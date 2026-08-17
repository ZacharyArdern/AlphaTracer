"""
Shared download utilities for AFDB and ESM Atlas structures and PAE matrices.
"""

import gzip as _gzip
import types as _types


def parse_fasta(path: str):
    """Yield simple records with .id and .seq for each entry in a FASTA file.

    Handles plain and gzip-compressed files. The .id is the first whitespace-
    delimited token of the header line (matching BioPython SeqIO behaviour).
    """
    open_fn = _gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rt') as fh:
        rec_id = seq_parts = None
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if rec_id is not None:
                    r = _types.SimpleNamespace(id=rec_id, seq=''.join(seq_parts))
                    yield r
                rec_id = line[1:].split()[0]
                seq_parts = []
            elif rec_id is not None:
                seq_parts.append(line)
        if rec_id is not None:
            yield _types.SimpleNamespace(id=rec_id, seq=''.join(seq_parts))
import os
import re
import sys
import asyncio
import io
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import numpy as np

AFDB_VERSION = 6

# ESM Atlas module directory — override with AT_ESM_DIR env var
_ESM_DIR = os.environ.get('AT_ESM_DIR', str(Path.home() / 'Science/Data/ESMAtlas'))


def get_afdb_id(sseqid: str) -> str | None:
    """Extract the AlphaFold accession from a sseqid field.

    Handles 'sp:AF-XXXX-F1', 'AF-XXXX-F1', and the common DIAMOND DB format
    where the version suffix is already present ('AF-XXXX-F1-model_v4').
    Always returns the bare accession WITHOUT the '-model_v4' version suffix
    so that afdb_url() and afdb_local_pdb() can append it consistently.

    >>> get_afdb_id('sp:AF-A0A000-F1')
    'AF-A0A000-F1'
    >>> get_afdb_id('AF-A0A000-F1')
    'AF-A0A000-F1'
    >>> get_afdb_id('AF-A0A000-F1-model_v4')
    'AF-A0A000-F1'
    >>> get_afdb_id('unknown') is None
    True
    """
    if sseqid.startswith('AF-'):
        acc = sseqid
    elif ':' in sseqid:
        # Handle 'sp:AF-XXXX-F1' or 'tr:AF-XXXX-F1' UniProt prefix formats
        acc = sseqid.split(':')[1]
        if not acc.startswith('AF-'):
            return None
    else:
        return None
    # Strip any version suffix already present (e.g. -model_v4, -model_v6)
    acc = re.sub(r'-model_v\d+$', '', acc)
    return acc


def afdb_url(afdb_id: str) -> str:
    return f'https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v{AFDB_VERSION}.pdb'

def afdb_local_pdb(afdb_id: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'{afdb_id}-model_v{AFDB_VERSION}.pdb')

def afdb_pae_local_path(afdb_id: str, pae_dir: str) -> str:
    return os.path.join(pae_dir, f'{afdb_id}-predicted_aligned_error_v{AFDB_VERSION}.json')

def afdb_pae_url(afdb_id: str) -> str:
    return (f'https://alphafold.ebi.ac.uk/files/'
            f'{afdb_id}-predicted_aligned_error_v{AFDB_VERSION}.json')

def esm_local_pdb(protein_hash: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'esm_{protein_hash}.pdb')

def is_valid_pdb(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            header = f.read(6).decode('ascii', errors='ignore')
        return header.strip()[:6] in ('HEADER', 'REMARK', 'ATOM  ', 'MODEL ')
    except Exception:
        return False


# ── AFDB PDB download ──────────────────────────────────────────────────────────

async def _fetch_afdb_pdbs_async(afdb_ids, pdb_dir: str) -> dict:
    sem = asyncio.Semaphore(64)
    connector = aiohttp.TCPConnector(limit=64)

    async def _fetch_one(session, aid):
        path = afdb_local_pdb(aid, pdb_dir)
        filename = os.path.basename(path)
        if os.path.exists(path) and is_valid_pdb(path):
            return aid, f'exists:{filename}'
        url = afdb_url(aid)
        async with sem:
            try:
                async with session.get(url) as resp:
                    data = await resp.read()
                with open(path, 'wb') as f:
                    f.write(data)
                if is_valid_pdb(path):
                    return aid, f'downloaded:{filename}'
                os.remove(path)
                return aid, f'failed:{aid}:server returned non-PDB content'
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)
                return aid, f'failed:{aid}:{e}'

    async with aiohttp.ClientSession(connector=connector) as session:
        return dict(await asyncio.gather(*[_fetch_one(session, aid) for aid in afdb_ids]))


def fetch_afdb_pdbs(afdb_ids, pdb_dir: str) -> dict:
    """Download AFDB PDB files with retry. Returns {aid: status_str}."""
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


# ── AFDB PAE download ──────────────────────────────────────────────────────────

def fetch_afdb_pae(afdb_ids, pae_dir: str) -> None:
    """Download AFDB PAE JSON files."""
    os.makedirs(pae_dir, exist_ok=True)
    afdb_ids = list(afdb_ids)
    missing = [aid for aid in sorted(afdb_ids)
               if not os.path.exists(afdb_pae_local_path(aid, pae_dir))]
    print(f'  {len(afdb_ids)} accessions; {len(missing)} PAE files to download')
    if not missing:
        return

    async def _fetch_all(aids):
        sem = asyncio.Semaphore(64)
        connector = aiohttp.TCPConnector(limit=64)

        async def _fetch_one(session, aid):
            path = afdb_pae_local_path(aid, pae_dir)
            url  = afdb_pae_url(aid)
            async with sem:
                for attempt in range(3):
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                            resp.raise_for_status()
                            data = await resp.read()
                        with open(path, 'wb') as f:
                            f.write(data)
                        return f'ok:{aid}'
                    except Exception as e:
                        if os.path.exists(path):
                            os.remove(path)
                        if attempt == 2:
                            return f'fail:{aid}:{e}'
                        await asyncio.sleep(0.5 * (attempt + 1))
            return f'fail:{aid}:unreachable'

        async with aiohttp.ClientSession(connector=connector) as session:
            return await asyncio.gather(*[_fetch_one(session, aid) for aid in aids])

    all_results = asyncio.run(_fetch_all(missing))
    n_ok = sum(1 for r in all_results if r.startswith('ok'))
    for r in all_results:
        if r.startswith('fail'):
            print(f'  PAE FAILED: {r}')
    print(f'  Downloaded: {n_ok}  Failed: {len(all_results) - n_ok}')


# ── ESM Atlas PDB + PAE download ───────────────────────────────────────────────

def fetch_esm_structures(esm_rows, pdb_dir: str, n_workers: int = 8,
                          pae_dir: str | None = None) -> None:
    """
    Fetch ESM Atlas PDB structures and optionally PAE matrices.

    esm_rows: iterable of dicts with keys protein_hash, and optionally
              fragment_id and frag_row. Pass polars rows via .to_dicts().
    pae_dir:  if given, PAE matrices are fetched simultaneously with structures.
    """
    esm_dir = os.path.abspath(_ESM_DIR)
    if esm_dir not in sys.path:
        sys.path.insert(0, esm_dir)
    try:
        import esm_query as _esm
    except ImportError:
        print(f'  [WARN] Cannot import esm_query from {esm_dir} — ESM Atlas hits will be skipped')
        return

    fetch_pae = pae_dir is not None
    hits, need_lookup, seen = [], [], set()
    hits_with_coords = []

    for row in esm_rows:
        ph  = row.get('protein_hash') or ''
        fid = row.get('fragment_id', -1)
        fr  = row.get('frag_row', -1)
        if not ph or ph in seen:
            continue
        seen.add(ph)
        needs_pdb = not os.path.exists(esm_local_pdb(ph, pdb_dir))
        needs_pae = fetch_pae and not os.path.exists(
            os.path.join(pae_dir, f'esm_{ph}.pae.npy'))
        if not needs_pdb and not needs_pae:
            continue
        if fid is not None and fid >= 0 and fr is not None and fr >= 0:
            h = {'fragment_id': fid, 'frag_row': fr, 'protein_hash': ph}
            if needs_pdb:
                hits.append(h)
            if needs_pae:
                hits_with_coords.append(h)
        else:
            need_lookup.append((ph, needs_pdb, needs_pae))

    if need_lookup:
        print(f'  Resolving {len(need_lookup)} ESM protein_hash(es) via local index...', flush=True)
        try:
            idx = _esm.lookup_hashes([x[0] for x in need_lookup])
            lookup_map = {r['protein_hash']: (r['fragment_id'], r['frag_row'])
                          for r in [dict(zip(idx.schema.names, row))
                                    for row in zip(*[idx.column(c).to_pylist()
                                                     for c in idx.schema.names])]}
            for ph, needs_pdb, needs_pae in need_lookup:
                coords = lookup_map.get(ph)
                if coords and coords[0] >= 0:
                    h = {'fragment_id': int(coords[0]), 'frag_row': int(coords[1]),
                         'protein_hash': ph}
                    if needs_pdb:
                        hits.append(h)
                    if needs_pae:
                        hits_with_coords.append(h)
                else:
                    print(f'  [WARN] No index entry for ESM protein_hash {ph} — skipping',
                          flush=True)
        except Exception as e:
            print(f'  [WARN] ESM lookup_hashes failed: {e}', flush=True)

    if not hits and not hits_with_coords:
        return
    if fetch_pae:
        os.makedirs(pae_dir, exist_ok=True)

    n_pdb_need, n_pae_need = len(hits), len(hits_with_coords)
    print(f'  Fetching {n_pdb_need} ESM Atlas PDB(s)'
          f'{f" + {n_pae_need} PAE(s)" if fetch_pae and n_pae_need else ""} from S3...',
          flush=True)
    t0 = time.time()

    def _do_pdb():
        if not hits:
            return None
        try:
            return _esm.query_from_hits(hits, columns=['protein_hash', 'structure_blob'],
                                        n_workers=n_workers)
        except Exception as e:
            print(f'  [WARN] ESM Atlas fetch failed: {type(e).__name__}: {e}')
            return None

    def _do_pae():
        if not fetch_pae or not hits_with_coords:
            return {}
        return _esm.fetch_pae_matrices(hits_with_coords, n_workers=n_workers)

    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_pdb = exe.submit(_do_pdb)
        fut_pae = exe.submit(_do_pae)
        pdb_table     = fut_pdb.result()
        pae_bytes_map = fut_pae.result()

    n_pdb = 0
    if pdb_table is not None:
        for batch in pdb_table.to_batches():
            for ph, blob in zip(batch['protein_hash'].to_pylist(),
                                batch['structure_blob'].to_pylist()):
                if blob is None:
                    continue
                pdb_path = esm_local_pdb(ph, pdb_dir)
                try:
                    with open(pdb_path, 'w') as f:
                        f.write(_esm.blob_to_pdb(blob))
                    n_pdb += 1
                except Exception as e:
                    print(f'  [WARN] ESM Atlas decode failed for {ph}: {e}')

    n_pae = 0
    for ph, pae_bytes in pae_bytes_map.items():
        npy_path = os.path.join(pae_dir, f'esm_{ph}.pae.npy')
        try:
            with zipfile.ZipFile(io.BytesIO(pae_bytes)) as zf:
                arr = np.load(io.BytesIO(zf.read(zf.namelist()[0])))
            np.save(npy_path, arr.astype(np.float32) / 8.0)
            n_pae += 1
        except Exception as e:
            print(f'  [WARN] ESM PAE decode failed for {ph}: {e}')

    msg = f'  Fetched {n_pdb}/{n_pdb_need} ESM structures'
    if fetch_pae:
        msg += f', {n_pae}/{n_pae_need} PAE matrices'
    print(msg + f' in {time.time()-t0:.1f}s', flush=True)
