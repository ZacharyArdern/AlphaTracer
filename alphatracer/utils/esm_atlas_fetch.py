"""
Download utilities for ESM Atlas structures and PAE matrices.

Includes the full ESM Atlas fetch engine (formerly esm_query.py):
three-tier blob fetch (blob cache → offset cache + HTTP range → Biohub API),
offset cache management, blob decoding, and PAE matrix fetch.
"""

import asyncio
import io
import json
import logging
import os
import struct
import subprocess
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import brotli
import msgpack
import numpy as np
import polars as pl
import requests
import zstd

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('RUST_LOG', 'error')
logging.getLogger('lance').setLevel(logging.ERROR)

try:
    import lance as _lance
    _HAS_LANCE = True
except ImportError:
    _lance = None
    _HAS_LANCE = False

# ── Path configuration ────────────────────────────────────────────────────────
# AT_ESM_DIR: root for blob cache, parquet index, sketch db
# AT_FRAG_CACHE_DIR: fragment offset cache (.npz) and frag_paths.json
#   defaults to AT_ESM_DIR/esm_atlas_fragment_cache/

_ESM_DIR = Path(os.environ.get('AT_ESM_DIR', str(Path.home() / 'Science/Data/ESMAtlas')))

_FRAG_CACHE_DIR = Path(os.environ.get(
    'AT_FRAG_CACHE_DIR',
    str(_ESM_DIR / 'esm_atlas_fragment_cache'),
))

_FRAG_PATHS_FILE = _FRAG_CACHE_DIR / 'frag_paths.json'
_INDEX_DIR       = _ESM_DIR / 'folds_1B_index' / 'merged'
_SKETCH_DB       = _ESM_DIR / 'esm_atlas_sketch_db.sidx'
_BLOB_CACHE_DIR  = _ESM_DIR / 'blob_cache'
_PAE_CACHE_DIR   = _FRAG_CACHE_DIR.parent / 'pae_cache'

_S3_HTTP         = 'https://esm-protein-atlas.s3.us-west-2.amazonaws.com/v1/folds/folds_1B.lance/data'
_S3_URI          = 's3://esm-protein-atlas/v1/folds'
_STORAGE_OPTIONS = {'aws_region': 'us-west-2', 'skip_signature': 'true'}
_BIOHUB_BASE     = 'https://biohub.ai'
_BIOHUB_BATCH_SIZE = 500
_ESM_CONCURRENCY = 128

_SEARCH_BIN = Path.home() / '.cache/alphatracer/sketch_rs/release/search'

# ── Per-process caches ────────────────────────────────────────────────────────

_dataset   = None
_frag_paths: dict[int, str] = {}


def _get_dataset():
    global _dataset
    if not _HAS_LANCE:
        raise ImportError('lance is not installed — required for non-structure_blob queries')
    if _dataset is None:
        _dataset = _lance.dataset(f'{_S3_URI}/folds_1B.lance',
                                  storage_options=_STORAGE_OPTIONS)
    return _dataset


def _get_frag_paths() -> dict[int, str]:
    global _frag_paths
    if not _frag_paths:
        if _FRAG_PATHS_FILE.exists():
            with open(_FRAG_PATHS_FILE) as f:
                _frag_paths = {int(k): v for k, v in json.load(f).items()}
        elif _HAS_LANCE:
            ds = _get_dataset()
            _frag_paths = {f.fragment_id: f.data_files()[0].path
                           for f in ds.get_fragments()}
        else:
            raise FileNotFoundError(
                f'frag_paths.json not found at {_FRAG_PATHS_FILE} and lance is not installed'
            )
    return _frag_paths


# ── Blob cache ────────────────────────────────────────────────────────────────

_PDB_MAGIC = (b'ATOM', b'REMA', b'MODE', b'HEAD', b'TITL', b'COMP', b'SEQR')


def _is_pdb_bytes(blob: bytes) -> bool:
    return blob[:4] in _PDB_MAGIC


def _blob_cache_path(protein_hash: str) -> Path:
    return _BLOB_CACHE_DIR / f'{protein_hash}.brotli'


def _pdb_cache_path(protein_hash: str) -> Path:
    return _BLOB_CACHE_DIR / f'{protein_hash}.pdb'


def _load_cached_blob(protein_hash: str) -> bytes | None:
    for p in (_blob_cache_path(protein_hash), _pdb_cache_path(protein_hash)):
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                pass
    return None


def _save_cached_blob(protein_hash: str, blob: bytes) -> None:
    _BLOB_CACHE_DIR.mkdir(exist_ok=True)
    target = _pdb_cache_path(protein_hash) if _is_pdb_bytes(blob) else _blob_cache_path(protein_hash)
    tmp = target.with_suffix('.tmp')
    tmp.write_bytes(blob)
    tmp.rename(target)


# ── Offset-index cache (FullZipLayout / MiniBlockLayout) ─────────────────────

def _pb_read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _pb_read_packed_varints(data: bytes) -> list[int]:
    vals, pos = [], 0
    while pos < len(data):
        v, pos = _pb_read_varint(data, pos)
        vals.append(v)
    return vals


def _pb_parse_fields(data: bytes) -> dict[int, list[tuple[int, object]]]:
    fields: dict[int, list] = {}
    pos = 0
    while pos < len(data):
        tag, pos = _pb_read_varint(data, pos)
        field_num, wire_type = tag >> 3, tag & 0x7
        if wire_type == 0:
            val, pos = _pb_read_varint(data, pos)
        elif wire_type == 2:
            length, pos = _pb_read_varint(data, pos)
            val = data[pos:pos + length]; pos += length
        elif wire_type == 1:
            val = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        else:
            break
        fields.setdefault(field_num, []).append((wire_type, val))
    return fields


def _download_frag_col_offsets(fid: int, col_idx: int) -> dict | None:
    """Shared tail-fetch + footer-parse + buf_offsets download logic.

    Returns a dict with keys: layout, fname, url, buf_pos, data_start, offsets.
    Returns None if the column's pages list is empty.
    """
    fname = _get_frag_paths()[fid]
    url   = f'{_S3_HTTP}/{fname}'

    _TAIL = 8192
    r1 = requests.get(url, headers={'Range': f'bytes=-{_TAIL}'}, timeout=30)
    r1.raise_for_status()
    file_size  = int(r1.headers['Content-Range'].split('/')[-1])
    tail_start = file_size - len(r1.content)
    tail       = r1.content

    def _tail_slice(offset: int, size: int) -> bytes:
        lo = offset - tail_start
        return tail[lo:lo + size]

    footer = tail[-40:]
    _, cmo_offset, gbo_offset, _, _, _, _ = struct.unpack_from('<QQQIIHHxxxx', footer)
    vals = struct.unpack_from(f'<{(gbo_offset-cmo_offset)//8}Q',
                              _tail_slice(cmo_offset, gbo_offset - cmo_offset))
    col_off, col_sz = vals[col_idx * 2], vals[col_idx * 2 + 1]
    col_bytes = _tail_slice(col_off, col_sz)
    top   = _pb_parse_fields(col_bytes)
    pages = top.get(2, [])

    if not pages:
        return None

    page_fields = _pb_parse_fields(pages[0][1])
    buf_offsets = _pb_read_packed_varints(page_fields[1][0][1]) if 1 in page_fields else []
    buf_sizes   = _pb_read_packed_varints(page_fields[2][0][1]) if 2 in page_fields else []

    if len(buf_offsets) == 2:
        buf0_pos  = buf_offsets[0]
        buf1_pos  = buf_offsets[1]
        buf1_size = buf_sizes[1]
        r = requests.get(url, headers={'Range': f'bytes={buf1_pos}-{buf1_pos+buf1_size-1}'}, timeout=60)
        r.raise_for_status()
        offsets = np.frombuffer(r.content, dtype=np.int32).copy()
        return {'layout': 'FullZip', 'fname': fname, 'url': url,
                'buf_pos': buf0_pos, 'data_start': 0, 'offsets': offsets}

    if len(buf_offsets) == 3:
        buf2_pos = buf_offsets[2]
        r_hdr = requests.get(url, headers={'Range': f'bytes={buf2_pos}-{buf2_pos+7}'}, timeout=30)
        r_hdr.raise_for_status()
        _, data_start = struct.unpack_from('<II', r_hdr.content)
        r_off = requests.get(url,
            headers={'Range': f'bytes={buf2_pos+8}-{buf2_pos+data_start-1}'}, timeout=60)
        r_off.raise_for_status()
        offsets = np.frombuffer(r_off.content, dtype=np.uint32).copy()
        return {'layout': 'MiniBlock', 'fname': fname, 'url': url,
                'buf_pos': buf2_pos, 'data_start': int(data_start), 'offsets': offsets}

    return None


def _build_frag_offset_cache(fid: int) -> dict | None:
    cache_file = _FRAG_CACHE_DIR / f'{fid}.npz'
    fname = _get_frag_paths()[fid]

    _FRAG_CACHE_DIR.mkdir(exist_ok=True)

    result = _download_frag_col_offsets(fid, col_idx=6)
    if result is None:
        return None

    layout   = result['layout']
    buf_pos  = result['buf_pos']
    data_start = result['data_start']
    offsets  = result['offsets']

    if layout == 'FullZip':
        np.savez_compressed(cache_file, layout=np.array('FullZip'),
                            buf0_pos=np.int64(buf_pos), offsets=offsets)
        return {'layout': 'FullZip', 'buf0_pos': buf_pos, 'fname': fname, 'offsets': offsets}

    np.savez_compressed(cache_file, layout=np.array('MiniBlock'),
                        buf2_pos=np.int64(buf_pos),
                        data_start=np.uint32(data_start), offsets=offsets)
    return {'layout': 'MiniBlock', 'buf2_pos': buf_pos, 'data_start': int(data_start),
            'fname': fname, 'offsets': offsets}


def _load_frag_cache(fid: int) -> dict | None:
    cache_file = _FRAG_CACHE_DIR / f'{fid}.npz'
    if not cache_file.exists():
        return None
    try:
        d = np.load(cache_file, allow_pickle=True)
        layout = str(d.get('layout', 'FullZip'))
        fname  = _get_frag_paths()[fid]
        if layout == 'FullZip':
            return {'layout': 'FullZip', 'buf0_pos': int(d['buf0_pos']),
                    'fname': fname, 'offsets': d['offsets']}
        if layout == 'MiniBlock' and 'buf2_pos' in d:
            return {'layout': 'MiniBlock', 'buf2_pos': int(d['buf2_pos']),
                    'data_start': int(d['data_start']),
                    'fname': fname, 'offsets': d['offsets']}
        return None
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None


def _blob_byte_range(cache: dict, frag_row: int) -> tuple[str, int, int]:
    offsets = cache['offsets']
    fname   = cache['fname']
    if cache['layout'] == 'FullZip':
        base  = cache['buf0_pos']
        start = base + int(offsets[frag_row])
        end   = base + int(offsets[frag_row + 1]) - 1
    else:
        base  = cache['buf2_pos'] + cache['data_start']
        start = base + int(offsets[frag_row])
        end   = base + int(offsets[frag_row + 1]) - 1
    return f'{_S3_HTTP}/{fname}', start, end


# ── PAE byte-range fetch ──────────────────────────────────────────────────────

def _build_frag_pae_cache(fid: int) -> dict | None:
    cache_file = _PAE_CACHE_DIR / f'{fid}.npz'
    if cache_file.exists():
        try:
            d = np.load(cache_file, allow_pickle=True)
            fname  = _get_frag_paths()[fid]
            layout = str(d.get('layout', np.array('FullZip')))
            return {'buf0_pos': int(d['buf0_pos']), 'fname': fname,
                    'offsets': d['offsets'], 'layout': layout}
        except Exception:
            cache_file.unlink(missing_ok=True)

    fname = _get_frag_paths()[fid]

    result = _download_frag_col_offsets(fid, col_idx=5)
    if result is None:
        return None

    layout     = result['layout']
    buf_pos    = result['buf_pos']
    data_start = result['data_start']
    offsets    = result['offsets']
    buf_data_pos = buf_pos + data_start

    _PAE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, layout=np.array(layout),
                        buf0_pos=np.int64(buf_data_pos), offsets=offsets)
    return {'buf0_pos': buf_data_pos, 'fname': fname, 'offsets': offsets, 'layout': layout}


async def _fetch_ranges_aiohttp(items, get_range, decompress) -> dict[str, bytes | None]:
    sem       = asyncio.Semaphore(_ESM_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=_ESM_CONCURRENCY)

    async def _fetch_one(session, h, cache):
        range_result = get_range(h, cache)
        if range_result is None:
            return h['protein_hash'], None
        url, start, end = range_result
        for attempt in range(3):
            try:
                async with sem:
                    async with session.get(
                        url, headers={'Range': f'bytes={start}-{end}'},
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        raw = await resp.read()
                blob = decompress(raw, cache)
                return h['protein_hash'], blob
            except Exception:
                if attempt == 2:
                    return h['protein_hash'], None
                await asyncio.sleep(2 ** attempt)
        return h['protein_hash'], None

    async with aiohttp.ClientSession(connector=connector) as session:
        pairs = await asyncio.gather(*[_fetch_one(session, h, cache) for h, cache in items])
    return dict(pairs)


# ── Module-level cache-build helpers ─────────────────────────────────────────

def _build_struct_cache_safe(fid):
    try:
        return fid, _build_frag_offset_cache(fid)
    except Exception:
        return fid, None


def _build_pae_cache_safe(fid):
    try:
        return fid, _build_frag_pae_cache(fid)
    except Exception:
        return fid, None


def fetch_pae_matrices(hits: list[dict], n_workers: int = 16) -> dict[str, bytes]:
    if not hits:
        return {}
    fids_needed = list({h['fragment_id'] for h in hits if h.get('fragment_id', -1) >= 0})

    pae_caches: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, len(fids_needed) or 1)) as exe:
        for fid, cache in exe.map(_build_pae_cache_safe, fids_needed):
            if cache is not None:
                pae_caches[fid] = cache

    items = [(h, pae_caches[h['fragment_id']])
             for h in hits
             if h.get('fragment_id', -1) >= 0 and h['fragment_id'] in pae_caches]
    if not items:
        return {}

    def _get_range_pae(h, cache):
        offsets = cache['offsets']
        base    = cache['buf0_pos']
        fr      = h['frag_row']
        if fr + 1 >= len(offsets):
            return None
        start = base + int(offsets[fr])
        end   = base + int(offsets[fr + 1]) - 1
        if start >= end:
            return None
        url = f'{_S3_HTTP}/{cache["fname"]}'
        return url, start, end

    def _decompress_pae(raw, cache):
        layout = cache.get('layout', 'FullZip')
        return zstd.decompress(raw[12:]) if layout == 'FullZip' else raw

    t0 = time.time()
    fetched = asyncio.run(_fetch_ranges_aiohttp(items, _get_range_pae, _decompress_pae))
    ok = sum(1 for v in fetched.values() if v is not None)
    print(f'    [ESM PAE] fetched {ok}/{len(items)} in {time.time()-t0:.1f}s', flush=True)
    return {ph: b for ph, b in fetched.items() if b is not None}


# ── Biohub API fallback ───────────────────────────────────────────────────────

def _fetch_biohub_batch(hashes: list[str], poll_interval: float = 1.0) -> dict[str, bytes]:
    r = requests.post(
        f'{_BIOHUB_BASE}/esm/protein/api/v1alpha1/proteins/batch',
        json={'protein_hashes': hashes, 'include_structure': True,
              'include_sequence': False, 'include_features': False},
        timeout=30,
    )
    r.raise_for_status()
    job      = r.json()
    job_id   = job.get('job_id') or job.get('id')
    poll_path = job.get('poll_url') or f'/esm/protein/api/v1alpha1/proteins/batch/jobs/{job_id}'
    poll_url  = poll_path if poll_path.startswith('http') else f'{_BIOHUB_BASE}{poll_path}'

    for _ in range(180):
        time.sleep(poll_interval)
        status_r = requests.get(poll_url, timeout=30)
        status_r.raise_for_status()
        status = status_r.json()
        if status.get('status') == 'completed':
            break
        if status.get('status') in ('failed', 'error'):
            raise RuntimeError(f'Biohub job {job_id} failed: {status}')
    else:
        raise TimeoutError(f'Biohub job {job_id} did not complete in 3 min')

    dl_url = status['download_url']
    if not dl_url.startswith('http'):
        dl_url = 'https:/' + dl_url
    dl = requests.get(dl_url, timeout=120)
    dl.raise_for_status()

    results: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(dl.content)) as z:
        for name in z.namelist():
            if name.startswith('structures/') and name.endswith('.pdb'):
                h = name[len('structures/'):-4]
                results[h] = z.read(name)
    return results


# ── Main blob fetch ───────────────────────────────────────────────────────────

def fetch_structure_blobs(hits: list[dict], n_workers: int = 16) -> dict[str, bytes]:
    """
    Fetch structure blobs for each hit. Returns {protein_hash: blob_bytes}.

    Tier 1: per-protein blob cache (instant)
    Tier 2: offset cache + single HTTP range request (~0.7s, parallel)
    Tier 3: build offset cache then HTTP; Biohub API fallback if cache build fails
    """
    if not hits:
        return {}

    results:    dict[str, bytes] = {}
    tier3_hits: list[dict]       = []

    for h in hits:
        blob = _load_cached_blob(h['protein_hash'])
        if blob is not None:
            results[h['protein_hash']] = blob
        else:
            tier3_hits.append(h)

    if not tier3_hits:
        return results

    tier2: list[tuple[dict, dict]] = []
    tier3: list[dict]              = []
    for h in tier3_hits:
        cache = _load_frag_cache(h['fragment_id'])
        if cache is not None:
            tier2.append((h, cache))
        else:
            tier3.append(h)

    def _get_range_blob(h, cache):
        url, start, end = _blob_byte_range(cache, h['frag_row'])
        return url, start, end

    def _decompress_blob(raw, cache):
        return zstd.decompress(raw[12:]) if cache['layout'] == 'FullZip' else raw

    if tier2:
        _t2 = time.time()
        fetched = asyncio.run(_fetch_ranges_aiohttp(tier2, _get_range_blob, _decompress_blob))
        t2_ok = 0
        for ph, blob in fetched.items():
            if blob is not None:
                _save_cached_blob(ph, blob)
                results[ph] = blob
                t2_ok += 1
        print(f'    [ESM fetch] tier2 (direct HTTP): {t2_ok}/{len(tier2)} in {time.time()-_t2:.1f}s', flush=True)

    if tier3:
        _t3 = time.time()
        fids_to_build = list({h['fragment_id'] for h in tier3})

        built: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=min(n_workers, len(fids_to_build) or 1)) as exe:
            for fid, cache in exe.map(_build_struct_cache_safe, fids_to_build):
                if cache is not None:
                    built[fid] = cache

        still_missing: list[dict]              = []
        tier3_direct:  list[tuple[dict, dict]] = []
        for h in tier3:
            cache = built.get(h['fragment_id'])
            if cache is not None:
                tier3_direct.append((h, cache))
            else:
                still_missing.append(h)

        t3_ok = 0
        if tier3_direct:
            fetched3 = asyncio.run(_fetch_ranges_aiohttp(tier3_direct, _get_range_blob, _decompress_blob))
            for ph, blob in fetched3.items():
                if blob is not None:
                    _save_cached_blob(ph, blob)
                    results[ph] = blob
                    t3_ok += 1

        print(f'    [ESM fetch] tier3 (direct HTTP): {t3_ok}/{len(tier3)} in {time.time()-_t3:.1f}s'
              f'  (built {len(built)} caches)', flush=True)

        if still_missing:
            _tb = time.time()
            tb_ok = 0
            hashes_missing = [h['protein_hash'] for h in still_missing]
            for i in range(0, len(hashes_missing), _BIOHUB_BATCH_SIZE):
                batch = hashes_missing[i:i + _BIOHUB_BATCH_SIZE]
                for attempt in range(3):
                    try:
                        for ph, pdb_bytes in _fetch_biohub_batch(batch).items():
                            results[ph] = pdb_bytes
                            _save_cached_blob(ph, pdb_bytes)
                            tb_ok += 1
                        break
                    except Exception as exc:
                        if attempt == 2:
                            print(f'    [ESM fetch] Biohub fallback failed: {exc}', flush=True)
                        else:
                            time.sleep(2 ** attempt)
            print(f'    [ESM fetch] Biohub fallback: {tb_ok}/{len(still_missing)} in {time.time()-_tb:.1f}s', flush=True)

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def lookup_hashes(hashes: list[str]) -> pl.DataFrame:
    """Map protein_hash strings to (fragment_id, frag_row) via local parquet index."""
    by_prefix = defaultdict(list)
    for h in hashes:
        by_prefix[h[0]].append(h)
    parts = []
    for prefix, hs in by_prefix.items():
        parquet = _INDEX_DIR / f'prefix_{prefix}.parquet'
        df = (pl.scan_parquet(parquet)
                .select(['protein_hash', 'fragment_id', 'frag_row'])
                .filter(pl.col('protein_hash').is_in(hs))
                .collect())
        parts.append(df)
    return pl.concat(parts) if parts else pl.DataFrame(
        schema={'protein_hash': pl.String, 'fragment_id': pl.Int32, 'frag_row': pl.Int32}
    )


def query_from_hits(hits: list[dict], columns: list[str] = None,
                    n_workers: int = 16) -> pl.DataFrame:
    """Fetch structure blobs for pre-parsed hits. Returns polars DataFrame."""
    if not hits:
        return pl.DataFrame()
    blobs = fetch_structure_blobs(hits, n_workers=n_workers)
    ph_list:   list[str]   = []
    blob_list: list[bytes] = []
    for h in hits:
        ph = h['protein_hash']
        if ph in blobs:
            ph_list.append(ph)
            blob_list.append(blobs[ph])
    df = pl.DataFrame({'protein_hash': ph_list, 'structure_blob': blob_list})
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df.select(keep)
    return df


_ATOM37_NAMES = [
    'N','CA','C','CB','O','CG','CG1','CG2','OG','OG1','SG',
    'CD','CD1','CD2','ND1','ND2','OD1','OD2','SD',
    'CE','CE1','CE2','CE3','NE','NE1','NE2','OE1','OE2',
    'CH2','NH1','NH2','OH','CZ','CZ2','CZ3','NZ','OXT'
]
_AA3 = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
    'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
    'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
    'W':'TRP','Y':'TYR','X':'UNK'
}


def _decode_ndarray(d: dict) -> np.ndarray:
    return np.frombuffer(d[b'data'], dtype=np.dtype(d[b'type'])).reshape(d[b'shape'])


def blob_to_pdb(blob: bytes, chain_id: str = 'A') -> str:
    """Decode an ESM Atlas structure blob to a PDB string."""
    if _is_pdb_bytes(blob):
        return blob.decode('utf-8', errors='replace')

    data  = msgpack.loads(brotli.decompress(blob), strict_map_key=False, raw=True)
    seq   = data[b'sequence'].decode()
    n_res = len(seq)

    mask_full = _decode_ndarray(data[b'atom37_mask']).reshape(n_res, 37).astype(bool)
    pos_flat  = _decode_ndarray(data[b'atom37_positions']).astype(np.float32)
    coords    = np.zeros((n_res, 37, 3), dtype=np.float32)
    coords[mask_full] = pos_flat

    if b'confidence' in data:
        conf_raw = _decode_ndarray(data[b'confidence']).ravel().astype(np.float32)
        plddt = conf_raw * 100.0 if conf_raw.max() <= 1.01 else conf_raw
    else:
        plddt = np.zeros(n_res, dtype=np.float32)

    if b'residue_index' in data and isinstance(data[b'residue_index'], dict):
        res_idx = _decode_ndarray(data[b'residue_index']).astype(int).ravel()
    else:
        res_idx = np.arange(1, n_res + 1)

    aa3_list = [_AA3.get(aa1, 'UNK') for aa1 in seq]
    lines = []
    for i in range(0, n_res, 13):
        chunk = aa3_list[i:i+13]
        lines.append(f"SEQRES {i//13+1:3d} {chain_id} {n_res:4d}  {' '.join(chunk)}\n")

    serial = 1
    for i, aa1 in enumerate(seq):
        aa3     = _AA3.get(aa1, 'UNK')
        res_num = int(res_idx[i]) if i < len(res_idx) else i + 1
        bfac    = float(plddt[i]) if i < len(plddt) else 0.0
        for j, atom_name in enumerate(_ATOM37_NAMES):
            if not mask_full[i, j]:
                continue
            x, y, z   = coords[i, j]
            name_col   = f' {atom_name:<3}' if len(atom_name) < 4 else atom_name[:4]
            lines.append(
                f"ATOM  {serial:5d} {name_col} {aa3} {chain_id}{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {atom_name[0]:>2}\n"
            )
            serial += 1
    lines.append("END\n")
    return ''.join(lines)


def search_sequences(
    query_fasta: str | Path,
    sidx_path: str | Path = None,
    top_k: int = 5,
    min_shared: int = 2,
    k: int = 9,
) -> list[dict]:
    """Run kmer sketch search against the ESM Atlas index."""
    sidx = Path(sidx_path) if sidx_path else _SKETCH_DB
    if not sidx.exists():
        raise FileNotFoundError(f'Sketch index not found: {sidx}')
    if not _SEARCH_BIN.exists():
        raise FileNotFoundError(f'search binary not found: {_SEARCH_BIN}')

    cmd  = [str(_SEARCH_BIN), str(sidx), str(query_fasta),
            str(top_k), str(min_shared), str(k)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    hits = []
    for line in proc.stdout.splitlines():
        if line.startswith('query\t') or not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        query_id, target_header, shared, jaccard = parts[:4]
        ph_parts = target_header.split('|')
        if len(ph_parts) < 3:
            continue
        try:
            protein_hash, fragment_id, frag_row = ph_parts[0], int(ph_parts[1]), int(ph_parts[2])
        except ValueError:
            continue
        hits.append({'query': query_id, 'protein_hash': protein_hash,
                     'fragment_id': fragment_id, 'frag_row': frag_row,
                     'shared': int(shared), 'jaccard': float(jaccard)})
    return hits


# ── AlphaTracer pipeline helpers ──────────────────────────────────────────────

def esm_local_pdb(protein_hash: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'esm_{protein_hash}.pdb')


def fetch_esm_structures(esm_rows, pdb_dir: str, n_workers: int = 8,
                          pae_dir: str | None = None,
                          frag_cache_dir: str | None = None) -> None:
    """
    Fetch ESM Atlas PDB structures and optionally PAE matrices.

    esm_rows:       iterable of dicts with keys protein_hash, and optionally
                    fragment_id and frag_row.
    pae_dir:        if given, PAE matrices are fetched simultaneously.
    frag_cache_dir: override _FRAG_CACHE_DIR (directory with .npz offset caches
                    and frag_paths.json). Defaults to the AT_FRAG_CACHE_DIR env
                    var or AT_ESM_DIR/esm_atlas_fragment_cache/.
    """
    global _FRAG_CACHE_DIR, _FRAG_PATHS_FILE
    if frag_cache_dir is not None:
        _FRAG_CACHE_DIR = Path(frag_cache_dir)
        _FRAG_PATHS_FILE = _FRAG_CACHE_DIR / 'frag_paths.json'
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
            idx = lookup_hashes([x[0] for x in need_lookup])
            lookup_map = {r['protein_hash']: (r['fragment_id'], r['frag_row'])
                          for r in idx.iter_rows(named=True)}
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
                    print(f'  [WARN] No index entry for ESM protein_hash {ph} — skipping', flush=True)
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
            return query_from_hits(hits, columns=['protein_hash', 'structure_blob'],
                                   n_workers=n_workers)
        except Exception as e:
            print(f'  [WARN] ESM Atlas fetch failed: {type(e).__name__}: {e}')
            return None

    def _do_pae():
        if not fetch_pae or not hits_with_coords:
            return {}
        return fetch_pae_matrices(hits_with_coords, n_workers=n_workers)

    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_pdb = exe.submit(_do_pdb)
        fut_pae = exe.submit(_do_pae)
        pdb_table     = fut_pdb.result()
        pae_bytes_map = fut_pae.result()

    n_pdb = 0
    if pdb_table is not None:
        for ph, blob in zip(pdb_table['protein_hash'].to_list(),
                            pdb_table['structure_blob'].to_list()):
            if blob is None:
                continue
            try:
                with open(esm_local_pdb(ph, pdb_dir), 'w') as f:
                    f.write(blob_to_pdb(blob))
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
