"""
Download utilities for AlphaFold Database (AFDB) structures and PAE matrices.
"""

import asyncio
import gzip
import os
import re
import types
from collections import Counter
from pathlib import Path

import aiohttp

AFDB_VERSION = 6
# Non-UniProt entries (zero-padded 16-digit numeric IDs like AF-0000000004497602) use v1
_NUMERIC_ID_RE = re.compile(r'^AF-\d{16}$')


def _model_version(afdb_id: str) -> int:
    """Return the correct model version for an AFDB accession."""
    return 1 if _NUMERIC_ID_RE.match(afdb_id) else AFDB_VERSION


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta(path: str):
    """Yield simple records with .id and .seq for each entry in a FASTA file.

    Handles plain and gzip-compressed files. The .id is the first
    whitespace-delimited token of the header line.
    """
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rt') as fh:
        rec_id = seq_parts = None
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if rec_id is not None:
                    yield types.SimpleNamespace(id=rec_id, seq=''.join(seq_parts))
                rec_id = line[1:].split()[0]
                seq_parts = []
            elif rec_id is not None:
                seq_parts.append(line)
        if rec_id is not None:
            yield types.SimpleNamespace(id=rec_id, seq=''.join(seq_parts))


# ── Path / URL helpers ────────────────────────────────────────────────────────

def get_afdb_id(sseqid: str) -> str | None:
    """Extract bare AF-XXXX-F1 accession from a sseqid field.

    Handles 'sp:AF-XXXX-F1', 'AF-XXXX-F1', and DIAMOND DB format with version
    suffix ('AF-XXXX-F1-model_v4'). Always strips the version suffix.

    >>> get_afdb_id('sp:AF-A0A000-F1')
    'AF-A0A000-F1'
    >>> get_afdb_id('AF-A0A000-F1-model_v4')
    'AF-A0A000-F1'
    >>> get_afdb_id('unknown') is None
    True
    """
    if sseqid.startswith('AF-'):
        acc = sseqid
    elif ':' in sseqid:
        acc = sseqid.split(':')[1]
        if not acc.startswith('AF-'):
            return None
    else:
        return None
    return re.sub(r'-model_v\d+$', '', acc)


def afdb_url(afdb_id: str) -> str:
    v = _model_version(afdb_id)
    return f'https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v{v}.pdb'

def afdb_local_pdb(afdb_id: str, pdb_dir: str) -> str:
    v = _model_version(afdb_id)
    return os.path.join(pdb_dir, f'{afdb_id}-model_v{v}.pdb')

def afdb_pae_local_path(afdb_id: str, pae_dir: str) -> str:
    v = _model_version(afdb_id)
    return os.path.join(pae_dir, f'{afdb_id}-predicted_aligned_error_v{v}.json')

def afdb_pae_url(afdb_id: str) -> str:
    v = _model_version(afdb_id)
    return (f'https://alphafold.ebi.ac.uk/files/'
            f'{afdb_id}-predicted_aligned_error_v{v}.json')


def is_valid_pdb(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            header = f.read(6).decode('ascii', errors='ignore')
        return header[:6].rstrip() in ('HEADER', 'REMARK', 'ATOM', 'MODEL')
    except Exception:
        return False


# ── PDB download ──────────────────────────────────────────────────────────────

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
                if data[:2] == b'\x1f\x8b':
                    data = gzip.decompress(data)
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
    """Download AFDB PDB files with one retry pass. Returns {aid: status_str}."""
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


# ── PAE download ──────────────────────────────────────────────────────────────

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
