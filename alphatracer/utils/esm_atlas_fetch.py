"""
Download utilities for ESM Atlas structures and PAE matrices.
"""

import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# ESM Atlas module directory — override with AT_ESM_DIR env var
_ESM_DIR = os.environ.get('AT_ESM_DIR', str(Path.home() / 'Science/Data/ESMAtlas'))


def esm_local_pdb(protein_hash: str, pdb_dir: str) -> str:
    return os.path.join(pdb_dir, f'esm_{protein_hash}.pdb')


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
        for ph, blob in zip(pdb_table['protein_hash'].to_list(),
                            pdb_table['structure_blob'].to_list()):
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
