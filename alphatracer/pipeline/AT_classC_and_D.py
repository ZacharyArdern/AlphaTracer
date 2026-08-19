#!/usr/bin/env python3
"""
AT_classC_and_D.py  —  AlphaTracer 1.0, Class C + D pipeline

Class C: domain-level matching for sequences not in Class A/B.
  Decomposes best-hit reference structures into rigid domains via PAE graph
  clustering.  Domains passing window-identity and beta-strand-indel checks
  are built with backbone grafting + CCD + OpenMM minimisation.
  Query regions outside the qualifying domain are filled using MLX MiniFold
  predictions + L-BFGS-B + CCD (use --no-fill-missing to disable).

Class D: full-length structure prediction for every query sequence not covered
  by Class A, B, or C.

  Recycling is pLDDT-gated: starts at 0 recyclings, retries up to --max-recyclings
  if mean pLDDT < --plddt-threshold.  If round-0 pLDDT < --min-recycle-plddt the
  sequence is accepted immediately (likely disordered; recycling won't help).

  On Apple Silicon (M-series Mac): MLX MiniFold is used for structure prediction.
  On Linux / WSL / Intel Mac: PyTorch MiniFold (jwohlwend/minifold) is used instead,
  requiring: pip install git+https://github.com/jwohlwend/minifold.git
  Backend can be overridden with --backend {auto|mlx|cuda|cpu}.

Usage
-----
  python AT_classC_and_D.py -i AT_processing_<name>/ [--min-pctsim 40] [-t 4]

  Weights are downloaded automatically from HuggingFace on first run.
"""

import math
import os
import re
import sys
import time
import json
import gzip
import argparse
import threading
import asyncio
import aiohttp
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
import gemmi
import parasail
from openmm import LangevinMiddleIntegrator, CustomExternalForce
from openmm.app import Simulation, PDBFile
import openmm.unit as unit
import platform as _platform

# ── Platform / backend detection ──────────────────────────────────────────────

def _is_apple_silicon():
    """True if running on Apple Silicon (M-series Mac)."""
    return _platform.system() == 'Darwin' and _platform.machine() == 'arm64'


def _resolve_backend(backend_arg):
    """Resolve --backend to bool: True → MLX, False → PyTorch (CUDA/CPU)."""
    if backend_arg == 'mlx':
        if not _is_apple_silicon():
            sys.exit('ERROR: --backend mlx requires Apple Silicon Mac')
        return True
    if backend_arg in ('cuda', 'cpu'):
        return False
    return _is_apple_silicon()  # 'auto'

# ── Local imports ─────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))   # alphatracer/
_PKG_PARENT  = os.path.dirname(_HERE)                        # repo root

# Ensure the repo root is on sys.path so 'alphatracer' is importable
# whether this file is run directly (subprocess) or as part of the package.
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from alphatracer.pipeline import AT_classA as _A
from alphatracer.pipeline import AT_classB as _B
from alphatracer.utils.pae_to_domains import (parse_pae_file, domains_from_pae_matrix_igraph,
                                               parse_pae_batch_rust, domains_from_pae_subsampled)
from alphatracer.utils.afdb_fetch import (
    AFDB_VERSION, afdb_local_pdb, afdb_pae_local_path, afdb_pae_url, fetch_afdb_pae,
)
from alphatracer.utils.esm_atlas_fetch import esm_local_pdb, _ESM_DIR, fetch_esm_structures

ONE_TO_THREE = _A.ONE_TO_THREE

# ── MiniFold-MLX weight paths (downloaded from HuggingFace on first use) ─────

_HF_REPO = 'z-ardern/MiniFold_MLX_weights'


def _get_weights(model_size='48L', use_quantized_esm=True):
    """Return (esm_path, minifold_path), downloading from HF if needed."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit('ERROR: huggingface_hub is required — pip install huggingface_hub')
    esm_folder  = 'ESM2_MiniFold_int8' if use_quantized_esm else 'ESM2_MiniFold'
    skip_folder = 'ESM2_MiniFold' if use_quantized_esm else 'ESM2_MiniFold_int8'
    print(f'  [MLX] Checking weights ({_HF_REPO}) …', flush=True)
    weights_dir = Path(snapshot_download(_HF_REPO, ignore_patterns=[f'{skip_folder}/*']))
    esm_path      = weights_dir / esm_folder
    minifold_path = weights_dir / f'minifold_{model_size}'
    if not esm_path.exists():
        sys.exit(f'ERROR: {esm_folder}/ not found in {weights_dir}')
    if not minifold_path.exists():
        sys.exit(f'ERROR: minifold_{model_size}/ not found in {weights_dir}')
    return str(esm_path), str(minifold_path)

# Global state — each loaded at most once
# MLX: (tokenizer, minifold_mlx, pad_id)
# PT:  (predict_mod, alphabet, model, config, device)  — PyTorch fallback
_MLX_STATE = None
_PT_STATE  = None
_USE_MLX   = None   # set on first _load_fold_models() call

import threading as _threading
_GPU_LOCK = _threading.Lock()  # serialise all MLX/GPU inference; C and D share this

# Fragment kmer search (set by main() when fill_missing is active)
_SEARCH_BIN        = None
_AFDB_SIDX_PATH    = None
_AFDB_PQ_PATH      = None
_ESM_SIDX_PATH     = None
_ESM_PQ_PATH       = None
_FRAG_PDB_DIR      = None   # local PDB cache dir for fragment template downloads
_FRAG_MIN_CONTAINMENT  = 0.08
_FRAG_MIN_IDENTITY = 0.40   # same as default Class B min_pctsim
_FRAG_MIN_COVERAGE = 0.70

_afdb_pq_cache = None
_esm_pq_cache  = None

_STATUS_PATH = None  # set in main() to proc_dir/.cd_status


def _write_status(msg: str) -> None:
    """Write a one-line status message for run_alphatracer.py status bar polling."""
    if _STATUS_PATH:
        try:
            with open(_STATUS_PATH, 'w') as _f:
                _f.write(msg)
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='AlphaTracer 1.0 — Class C + D pipeline'
    )
    p.add_argument('-i', '--input-dir', required=True,
                   help='Processing directory from AT_classA.py / AT_classB.py')
    p.add_argument('-t', '--threads',       type=int,   default=4)
    # Class C options
    p.add_argument('--min-pctsim',           type=float, default=40.0)
    p.add_argument('--window-size',         type=int,   default=40)
    p.add_argument('--pae-cutoff',          type=float, default=5.0)
    p.add_argument('--pae-power',           type=float, default=1.0)
    p.add_argument('--pae-resolution',      type=float, default=1.0)
    p.add_argument('--pae-step',            type=int,   default=0,
                   help='Subsampling step for PAE matrix (1=full, 2=every 2nd, etc). '
                        '0=auto: step=2 for n≤300, step=3 for n>300 (default: 0)')

    p.add_argument('--pae-resolution-large', type=float, default=2.0,
                   help='graph_resolution for domains >--large-domain-threshold aa (default: 2.0)')
    p.add_argument('--large-domain-threshold', type=int, default=500,
                   help='Residue count above which --pae-resolution-large is used (default: 500)')
    p.add_argument('--min-domain-size',      type=int,   default=30,
                   help='Minimum domain residue count for Class C; smaller domains go to Class D (default: 30)')
    p.add_argument('--min-domain-plddt',     type=float, default=70.0,
                   help='Minimum mean AF pLDDT over domain residues for Class C; low-confidence domains go to Class D (default: 70)')
    p.add_argument('--mm-iters',            type=int,   default=300)
    p.add_argument('--ccd-iters',           type=int,   default=200)
    p.add_argument('--ccd-tol',             type=float, default=0.15)
    p.add_argument('--flank',               type=int,   default=3)
    p.add_argument('--limit',               type=int,   default=0)
    p.add_argument('--classC-top-k',        type=int,   default=100,
                   help='Max hits per query to NW-align before selecting best by coverage (default: 100)')
    p.add_argument('--no-fill-missing',     action='store_false', dest='fill_missing', default=True,
                   help='Fill non-domain regions with MLX MiniFold (classC)')
    p.add_argument('--min-frag-len',        type=int,   default=5)
    p.add_argument('--lbfgsb-iters',        type=int,   default=300)
    p.add_argument('--mini-model-size',     type=str,   default='12L',
                   choices=['12L', '48L'])
    p.add_argument('--anchor-k',            type=float, default=1000.0,
                   help='kJ/mol/nm² for segment-centre Cα anchor restraints')
    # Recycling options
    p.add_argument('--plddt-threshold',     type=float, default=85.0,
                   help='pLDDT threshold: recycle until mean pLDDT >= this (default: 85)')
    p.add_argument('--max-recyclings',      type=int,   default=2,
                   help='Maximum recycling rounds (default: 2)')
    p.add_argument('--min-recycle-plddt',   type=float, default=50.0,
                   help='Stop recycling if pLDDT after round 0 is below this — '
                        'sequence is likely disordered (default: 50)')
    # Class D options
    p.add_argument('--no-write-candidates',  action='store_false', dest='write_candidates', default=True,
                   help='Skip writing classC_candidates.pq (alignment + outcome for every Class C candidate)')
    p.add_argument('--no-classD',           action='store_true', default=False,
                   help='Skip Class D predictions')
    p.add_argument('--classD-limit',        type=int,   default=0,
                   help='Limit Class D to first N sequences (0 = no limit)')
    p.add_argument('--batch-tokens',        type=int,   default=262144,
                   help='Max L² token budget per batch in classD (default: 262144). '
                        'Allows sequences up to ~512 aa to batch together.')
    p.add_argument('--no-compile',          action='store_true', default=False,
                   help='Disable mx.compile on MiniFormer (MLX backend only)')
    p.add_argument('--max-seq-len',         type=int,   default=1000,
                   help='Skip MLX prediction for sequences longer than this (default: 1000). '
                        'ESM2-3B attention maps scale as L², causing Metal OOM for long seqs. '
                        'Set to 0 to disable the check.')
    p.add_argument('--loop-closer',         default='ccd', choices=['ccd', 'promod3'],
                   help='Loop closing backend for Class C: ccd (default) or promod3')
    p.add_argument('--promod3-data-dir',    default=None,
                   help='ProMod3 database directory (overrides PROMOD3_SHARED_DATA_PATH)')
    p.add_argument('--backend',             default='auto',
                   choices=['auto', 'mlx', 'cuda', 'cpu'],
                   help='Inference backend: auto (detect; MLX on Apple Silicon, '
                        'PyTorch on Linux/Intel Mac), mlx, cuda, or cpu')
    return p.parse_args()


# ── ESM fragment index lookup ─────────────────────────────────────────────────

def _resolve_esm_fragment_ids(rows):
    """Populate fragment_id/frag_row for ESM rows where they are -1 (diamond hits).
    Looks up by first 32 chars of protein_hash in the local folds_1B_index."""
    needs = [r for r in rows if r.get('fragment_id', -1) == -1 and r.get('protein_hash')]
    if not needs:
        return
    import glob as _glob
    esm_dir = os.path.abspath(_ESM_DIR)
    index_dir = os.path.join(esm_dir, 'folds_1B_index', 'merged')
    if not os.path.isdir(index_dir):
        print(f'  [WARN] ESM fragment index not found at {index_dir} — PAE unavailable for diamond ESM hits')
        return
    # Group by prefix (first hex char of protein_hash) to read only needed files
    by_prefix = {}
    for r in needs:
        ph = r['protein_hash'][:32]
        by_prefix.setdefault(ph[0], []).append((r, ph))
    ph_to_frag = {}
    for prefix, entries in by_prefix.items():
        pq_path = os.path.join(index_dir, f'prefix_{prefix}.parquet')
        if not os.path.exists(pq_path):
            continue
        ph_set = {ph for _, ph in entries}
        idx = pl.read_parquet(pq_path, columns=['protein_hash', 'fragment_id', 'frag_row'])
        idx = idx.filter(pl.col('protein_hash').is_in(ph_set))
        for row in idx.iter_rows(named=True):
            ph_to_frag[row['protein_hash']] = (row['fragment_id'], row['frag_row'])
    resolved = 0
    for r, ph in [(r, r['protein_hash'][:32]) for r in needs]:
        if ph in ph_to_frag:
            r['fragment_id'], r['frag_row'] = ph_to_frag[ph]
            resolved += 1
    print(f'  [ESM index] Resolved fragment_id/frag_row for {resolved}/{len(needs)} diamond ESM hits')


# ── PAE helpers ───────────────────────────────────────────────────────────────

# _pae_local_path, _pae_url, stage_download_pae removed — use
# afdb_pae_local_path, afdb_pae_url, fetch_afdb_pae from utils.afdb_fetch

# Alias for internal references still using the old name
_pae_local_path = afdb_pae_local_path


def _load_pae_matrix(pae_path):
    opener = gzip.open if pae_path.endswith('.gz') else open
    with opener(pae_path, 'rt') as f:
        raw = json.load(f)
    data = raw[0] if isinstance(raw, list) else raw
    if 'predicted_aligned_error' in data:
        return np.array(data['predicted_aligned_error'], dtype=np.float64)
    if 'residue1' in data and 'distance' in data:
        r1, d = data['residue1'], data['distance']
        size = max(r1)
        m = np.empty((size, size), dtype=np.float64)
        m.ravel()[:] = d
        return m
    raise ValueError(f'Unrecognised PAE format in {pae_path}')


def get_domains(pae_path, pae_power, pae_cutoff, pae_resolution):
    try:
        pae = _load_pae_matrix(pae_path)
        domains = domains_from_pae_matrix_igraph(
            pae, pae_power=pae_power, pae_cutoff=pae_cutoff,
            graph_resolution=pae_resolution,
        )
        return [sorted(d) for d in domains]
    except Exception as e:
        print(f'  PAE domain error ({pae_path}): {e}')
        return []


def get_domains_from_matrix(pae_matrix, n_full, step, pae_power, pae_cutoff, pae_resolution):
    """Domain detection on a pre-parsed (possibly subsampled) PAE matrix."""
    try:
        if step > 1:
            return domains_from_pae_subsampled(
                pae_matrix, n_full, step,
                pae_power=pae_power, pae_cutoff=pae_cutoff,
                graph_resolution=pae_resolution,
            )
        domains = domains_from_pae_matrix_igraph(
            pae_matrix, pae_power=pae_power, pae_cutoff=pae_cutoff,
            graph_resolution=pae_resolution,
        )
        return [sorted(d) for d in domains]
    except Exception as e:
        print(f'  PAE domain error (pre-parsed matrix): {e}')
        return []


# ── Alignment helpers ─────────────────────────────────────────────────────────

def _ref_pos_to_comp(qseq_alg, sseq_alg, alg_comp):
    q_start    = len(qseq_alg) - len(qseq_alg.lstrip('-'))
    q_end      = len(qseq_alg.rstrip('-')) or len(qseq_alg)
    ref_offset = sum(1 for c in sseq_alg[:q_start] if c != '-')
    rp = ref_offset
    mapping = {}
    for i in range(q_start, q_end):
        q, s = qseq_alg[i], sseq_alg[i]
        if q != '-' and s != '-':
            mapping[rp] = alg_comp[i]
            rp += 1
        elif q == '-':
            rp += 1
    return mapping


def _domain_ops(domain_set, ops):
    result = []
    for op in ops:
        if op[0] == 'match':
            if op[1] in domain_set:
                result.append(op)
        elif op[0] == 'deletion':
            _, ds, count = op
            if any((ds + j) in domain_set for j in range(count)):
                result.append(op)
        elif op[0] == 'insertion':
            if op[2] in domain_set or op[3] in domain_set:
                result.append(op)
    return result


def _domain_comp_str(domain_set, d_ops, rp_to_comp):
    chars = []
    for op in d_ops:
        if op[0] == 'match':
            _, rpos, _ = op
            if rpos in domain_set:
                chars.append(rp_to_comp.get(rpos, ':'))
        elif op[0] == 'deletion':
            _, ds, count = op
            for j in range(count):
                if (ds + j) in domain_set:
                    chars.append('-')
        elif op[0] == 'insertion':
            chars.extend(['-'] * len(op[1]))
    return ''.join(chars)


# ── Domain classification ─────────────────────────────────────────────────────

def classify_domain(domain_indices, ops, ref_ss, rp_to_comp, window, threshold_frac):
    if not domain_indices:
        return False
    domain_set = set(domain_indices)
    d_ops = _domain_ops(domain_set, ops)

    for k, op in enumerate(d_ops):
        if op[0] in ('deletion', 'insertion'):
            if _B._indel_ss_context(k, d_ops, ref_ss) == 'E':
                return False

    comp = _domain_comp_str(domain_set, d_ops, rp_to_comp)
    return _A.all_windows_pass(comp, window, threshold_frac)


# ── Anchor-restraint helpers ──────────────────────────────────────────────────

def _find_segment_anchors(residues, ref_poly):
    anchors = []
    i, n = 0, len(residues)
    while i < n:
        res = residues[i]
        if not (res.get('from_ref') and 'CA' in res['atoms'] and 'ref_pos' in res):
            i += 1
            continue
        seg = []
        j = i
        while (j < n and residues[j].get('from_ref')
               and 'CA' in residues[j]['atoms'] and 'ref_pos' in residues[j]):
            seg.append(j)
            j += 1
        if seg:
            ca_pos  = np.array([residues[ri]['atoms']['CA'] for ri in seg])
            centre  = ca_pos.mean(0)
            dists   = np.linalg.norm(ca_pos - centre, axis=1)
            best_ri = seg[int(np.argmin(dists))]
            rpos    = residues[best_ri]['ref_pos']
            if rpos < len(ref_poly):
                a = ref_poly[rpos].find_atom('CA', '\0')
                if a:
                    ref_xyz = np.array([a.pos.x, a.pos.y, a.pos.z])
                    anchors.append((best_ri, ref_xyz))
        i = j
    return anchors


def _ca_flat_index(residues, target_ri):
    flat = 0
    for ri, res in enumerate(residues):
        for aname in ('N', 'CA', 'C', 'O', 'CB'):
            if aname in res['atoms']:
                if ri == target_ri and aname == 'CA':
                    return flat
                flat += 1
    return None


def _add_anchor_restraints(system, residues, anchors, k_kj_mol_nm2=1000.0):
    if not anchors:
        return
    force = CustomExternalForce('k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    force.addGlobalParameter('k', k_kj_mol_nm2)
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')
    for (ri, ref_xyz) in anchors:
        flat_idx = _ca_flat_index(residues, ri)
        if flat_idx is not None:
            force.addParticle(flat_idx,
                              [ref_xyz[0] / 10.0,
                               ref_xyz[1] / 10.0,
                               ref_xyz[2] / 10.0])
    system.addForce(force)


# ── Domain PDB builder ────────────────────────────────────────────────────────

def build_domain_pdb(domain_indices, ops, ref_poly, out_pdb,
                     mm_iters, ccd_iters, ccd_tol, n_flank,
                     anchor_k=1000.0, loop_closer='ccd'):
    t0 = time.perf_counter()
    try:
        domain_set = set(domain_indices)
        ref_ss     = _B.compute_ref_ss(ref_poly)
        d_ops      = _domain_ops(domain_set, ops)

        if not d_ops:
            return False, 'no ops in domain', 0.

        indel_contexts = {}
        for k, op in enumerate(d_ops):
            if op[0] in ('deletion', 'insertion'):
                indel_contexts[k] = _B._indel_ss_context(k, d_ops, ref_ss)

        def ref_backbone(res):
            atoms = {}
            for aname in ['N', 'CA', 'C', 'O', 'CB']:
                try:
                    a = res.find_atom(aname, '\0')
                    if a:
                        atoms[aname] = np.array([a.pos.x, a.pos.y, a.pos.z])
                except Exception:
                    pass
            return atoms

        residues  = []
        gap_sites = []

        for op_idx, op in enumerate(d_ops):
            if op[0] == 'match':
                _, rpos, qaa = op
                if rpos >= len(ref_poly):
                    return False, f'ref_pos {rpos} out of range', 0.
                resname = ONE_TO_THREE.get(qaa, 'ALA')
                atoms   = ref_backbone(ref_poly[rpos])
                keep    = {'N', 'CA', 'C', 'O'} | ({'CB'} if qaa != 'G' else set())
                atoms   = {k: v for k, v in atoms.items() if k in keep}
                if len({'N', 'CA', 'C', 'O'} & atoms.keys()) < 4:
                    return False, f'incomplete backbone at ref_pos {rpos}', 0.
                residues.append({'resname': resname, 'atoms': atoms,
                                 'from_ref': True, 'ref_pos': rpos})

            elif op[0] == 'deletion':
                if residues:
                    gap_sites.append((len(residues) - 1, 'deletion'))

            elif op[0] == 'insertion':
                ins_aas = op[1]
                if not residues:
                    continue  # leading insertion before any matched residues — skip, same as leading deletion
                gap_sites.append((len(residues) - 1, 'insertion'))
                prev = residues[-1]['atoms']
                if not all(k in prev for k in ('N', 'CA', 'C')):
                    return False, 'cannot place insertion: prev residue missing backbone', 0.
                pN, pCA, pC = prev['N'], prev['CA'], prev['C']
                ss_ctx  = indel_contexts.get(op_idx, 'C')
                builder = (_B.build_helix_residue if ss_ctx == 'H'
                           else _B.build_extended_residue)
                for aa in ins_aas:
                    resname   = ONE_TO_THREE.get(aa, 'ALA')
                    new_atoms = builder(pN, pCA, pC, resname)
                    residues.append({'resname': resname, 'atoms': new_atoms,
                                     'from_ref': False})
                    pN, pCA, pC = new_atoms['N'], new_atoms['CA'], new_atoms['C']

        if not residues:
            return False, 'no residues built', 0.

        if gap_sites:
            flat_atoms  = []
            flat_pos    = []
            atom_to_idx = {}
            for ri, res in enumerate(residues):
                for aname in ['N', 'CA', 'C', 'O', 'CB']:
                    if aname in res['atoms']:
                        atom_to_idx[(ri, aname)] = len(flat_atoms)
                        flat_atoms.append((ri, aname))
                        flat_pos.append(res['atoms'][aname].copy())

            for (gap_before_ri, gap_type) in gap_sites:
                after_ri   = gap_before_ri + 1
                if after_ri >= len(residues):
                    continue
                target_idx = atom_to_idx.get((after_ri, 'N'))
                if target_idx is None:
                    continue
                target  = flat_pos[target_idx].copy()
                end_ri  = (after_ri - 1 if gap_type == 'insertion'
                           else gap_before_ri)
                end_idx = atom_to_idx.get((end_ri, 'C'))
                if end_idx is None:
                    continue
                if gap_type == 'insertion':
                    mobile_start = max(0, gap_before_ri + 1 - n_flank)
                    mobile_end   = after_ri
                else:
                    mobile_start = max(0, gap_before_ri + 1 - n_flank)
                    mobile_end   = gap_before_ri + 1

                pivot_bonds = []
                for ri in range(mobile_start, mobile_end):
                    for (a_name, b_name) in [('N', 'CA'), ('CA', 'C')]:
                        ia = atom_to_idx.get((ri, a_name))
                        ib = atom_to_idx.get((ri, b_name))
                        if ia is None or ib is None:
                            continue
                        downstream = [
                            atom_to_idx[(rj, an)]
                            for rj in range(ri, mobile_end)
                            for an in ['N', 'CA', 'C', 'O', 'CB']
                            if (rj, an) in atom_to_idx
                            and atom_to_idx[(rj, an)] > ib
                        ]
                        if end_idx > ib and end_idx not in downstream:
                            downstream.append(end_idx)
                        if downstream:
                            pivot_bonds.append((ia, ib, downstream))

                if pivot_bonds:
                    _B.ccd_close(flat_pos, pivot_bonds, end_idx, target,
                                 max_iter=ccd_iters, tol=ccd_tol)

            for (ri, aname), pos in zip(flat_atoms, flat_pos):
                residues[ri]['atoms'][aname] = pos

        # ── ProMod3 loop closing (replaces CCD geometry when requested) ────────
        if loop_closer == 'promod3' and gap_sites:
            from alphatracer.utils.loop_closer import close_gap_promod3
            trg_seq, tpl_seq = [], []
            for op in d_ops:
                if op[0] == 'match':
                    _, rpos, qaa = op
                    trg_seq.append(qaa); tpl_seq.append(qaa)
                elif op[0] == 'insertion':
                    ins_aas = op[1]
                    trg_seq.append(ins_aas)
                    tpl_seq.append('-' * len(ins_aas))
                elif op[0] == 'deletion':
                    _, ds, count = op
                    trg_seq.append('-' * count)
                    for j in range(count):
                        rp = ds + j
                        aa3 = ref_poly[rp].name if rp < len(ref_poly) else 'ALA'
                        tpl_seq.append(_B.THREE_TO_ONE.get(aa3, 'X'))
            trg_str = ''.join(trg_seq).replace('-', '')
            tpl_str = ''.join(tpl_seq)

            _st = gemmi.Structure()
            _mdl = gemmi.Model('1')
            _chn = gemmi.Chain('A')
            for ri, res_d in enumerate(residues):
                _res = gemmi.Residue()
                _res.name = res_d.get('resname', 'ALA')
                _res.seqid = gemmi.SeqId(ri + 1, ' ')
                for aname, xyz in res_d.get('atoms', {}).items():
                    _atom = gemmi.Atom()
                    _atom.name = aname
                    _atom.pos = gemmi.Position(*xyz)
                    _atom.element = gemmi.Element(aname[0])
                    _res.add_atom(_atom)
                _chn.add_residue(_res)
            _mdl.add_chain(_chn)
            _st.add_model(_mdl)

            _filled, _err = close_gap_promod3(
                _st, chain_id='A',
                n_stem_seqid=None, c_stem_seqid=None,
                gap_seq=None, target_seq=trg_str, tpl_seq_with_gaps=tpl_str)
            if _err is None and _filled is not None:
                for _res in _filled[0]['A']:
                    _ri = _res.seqid.num - 1
                    if 0 <= _ri < len(residues):
                        for _atom in _res:
                            residues[_ri]['atoms'][_atom.name] = [
                                _atom.pos.x, _atom.pos.y, _atom.pos.z]

        anchors = _find_segment_anchors(residues, ref_poly)
        system, top, pos_nm = _B.build_openmm_system(residues)
        _add_anchor_restraints(system, residues, anchors, anchor_k)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
        )
        sim = Simulation(top, system, integrator, platform=_B._get_platform())
        sim.context.setPositions(pos_nm)
        sim.minimizeEnergy(maxIterations=mm_iters)
        state = sim.context.getState(getPositions=True)
        with open(out_pdb, 'w') as f:
            PDBFile.writeFile(top, state.getPositions(), f)

        return True, None, time.perf_counter() - t0

    except Exception as e:
        return False, str(e), time.perf_counter() - t0


def _split_broken_chains(pdb_path, ca_gap_threshold=5.5):
    """Re-label chain IDs in a PDB so that each contiguous segment gets its own
    chain letter.  A new chain starts wherever consecutive CA atoms are more than
    *ca_gap_threshold* Å apart (default 5.5 Å; a normal peptide CA-CA is ~3.8 Å,
    so a single missing residue already exceeds this).  Operates in-place."""
    import string
    chain_letters = list(string.ascii_uppercase)

    with open(pdb_path) as fh:
        lines = fh.readlines()

    # Collect CA positions in order of appearance
    ca_positions = []  # list of (line_index, x, y, z)
    for i, line in enumerate(lines):
        if line[:6] in ('ATOM  ', 'HETATM') and line[12:16].strip() == 'CA':
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_positions.append((i, x, y, z))
            except ValueError:
                pass

    if len(ca_positions) < 2:
        return  # nothing to split

    # Assign a segment index to each CA
    seg = 0
    ca_seg = [0]
    for k in range(1, len(ca_positions)):
        _, x0, y0, z0 = ca_positions[k - 1]
        _, x1, y1, z1 = ca_positions[k]
        d = ((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) ** 0.5
        if d > ca_gap_threshold:
            seg += 1
        ca_seg.append(seg)

    if seg == 0:
        return  # no breaks found, nothing to do

    # Build mapping: line_index → new chain letter
    line_to_chain = {}
    for (line_idx, *_), s in zip(ca_positions, ca_seg):
        line_to_chain[line_idx] = chain_letters[min(s, len(chain_letters) - 1)]

    # For non-CA ATOM lines, inherit chain from the nearest preceding CA in same residue
    # Strategy: scan sequentially, track current chain from last seen CA line
    current_chain = chain_letters[0]
    new_lines = []
    ca_iter = iter(zip([li for li, *_ in ca_positions], ca_seg))
    next_ca_line, next_ca_seg = next(ca_iter, (None, 0))

    for i, line in enumerate(lines):
        if line[:6] in ('ATOM  ', 'HETATM'):
            if i == next_ca_line:
                current_chain = chain_letters[min(next_ca_seg, len(chain_letters) - 1)]
                next_ca_line, next_ca_seg = next(ca_iter, (None, 0))
            line = line[:21] + current_chain + line[22:]
        elif line.startswith('TER'):
            line = line[:21] + current_chain + line[22:] if len(line) > 21 else line
        new_lines.append(line)

    # Insert TER records between chain segments
    output = []
    prev_chain = None
    for line in new_lines:
        if line[:6] in ('ATOM  ', 'HETATM'):
            ch = line[21]
            if prev_chain is not None and ch != prev_chain:
                output.append(f'TER\n')
            prev_chain = ch
        output.append(line)

    with open(pdb_path, 'w') as fh:
        fh.writelines(output)


# ── MLX model loading and prediction ─────────────────────────────────────────

def _load_mlx_models(model_size='12L', compile_miniformer=True):
    """Load MLX MiniFold (with fine-tuned ESM2 layers) once."""
    global _MLX_STATE
    if _MLX_STATE is not None:
        return

    _write_status('Loading MLX model (ESM2-3B + MiniFold)...')
    print('  [MLX] Loading MLX ESM2-3B + MiniFold (once)...', flush=True)
    t0 = time.perf_counter()

    import mlx.core as mx
    import mlx.nn as nn_mlx
    from minifold_mlx.esm2 import ESM2
    from minifold_mlx import MiniFoldMLX

    esm_path, minifold_path = _get_weights(model_size, use_quantized_esm=True)

    tokenizer, esm_model = ESM2.from_pretrained(esm_path)
    mx.eval(esm_model.parameters())

    minifold_mlx = MiniFoldMLX.from_pretrained(minifold_path, esm_model=esm_model)
    minifold_mlx.convert_to_bf16()

    if compile_miniformer and hasattr(minifold_mlx, 'enable_compile_miniformer'):
        minifold_mlx.enable_compile_miniformer()
    elif compile_miniformer:
        print('  [MLX] enable_compile_miniformer not available — skipping', flush=True)

    pad_id = int(getattr(tokenizer, 'pad_id', 1))

    _MLX_STATE = (tokenizer, minifold_mlx, pad_id)
    print(f'  [MLX] Ready in {time.perf_counter()-t0:.1f}s', flush=True)


def _mask_selenocysteine(seq):
    """Replace U (selenocysteine) with C (cysteine); return (masked_seq, sec_positions).

    sec_positions is a list of 0-based indices into seq where U appeared.
    MiniFold has no SEC token, so prediction must use CYS; we restore SEC afterwards.
    """
    if 'U' not in seq:
        return seq, []
    positions = [i for i, aa in enumerate(seq) if aa == 'U']
    masked = seq.replace('U', 'C')
    return masked, positions


def _restore_selenocysteine(pdb_str, sec_positions):
    """Post-process PDB string to restore SEC residues at sec_positions (0-based seq indices).

    For each SEC position: rename residue name CYS→SEC and atom name SG→SE.
    The residue sequence number in the PDB is 1-based (position+1).
    """
    if not sec_positions:
        return pdb_str
    sec_resnums = {p + 1 for p in sec_positions}
    lines = []
    for line in pdb_str.splitlines(keepends=True):
        if line[:6] in ('ATOM  ', 'HETATM'):
            try:
                resnum = int(line[22:26])
            except ValueError:
                lines.append(line)
                continue
            if resnum in sec_resnums:
                # rename residue CYS → SEC (cols 17-19)
                line = line[:17] + 'SEC' + line[20:]
                # rename sidechain SG atom → SE  (cols 12-15)
                atom_name = line[12:16]
                if atom_name.strip() == 'SG':
                    line = line[:12] + ' SE ' + line[16:]
        lines.append(line)
    return ''.join(lines)




def _mlx_predict(seq, num_recycling):
    """Run MLX MiniFold on seq (no TTT).
    Returns (pdb_str, mean_plddt, elapsed_s).
    """
    global _MLX_STATE
    import mlx.core as mx
    from minifold_mlx._data import prepare_input, output_to_pdb

    t0 = time.perf_counter()
    tokenizer, minifold_mlx, pad_id = _MLX_STATE

    tokens, mask_np, aatype = prepare_input(seq, tokenizer)
    L = len(seq)

    tokens_mx = mx.array(np.array(tokens))[None]   # (1, L)
    mask_mx   = mx.array(mask_np)[None]             # (1, L)
    aatype_mx = mx.array(aatype)[None]              # (1, L)

    out = minifold_mlx(tokens_mx, seq_mask=mask_mx,
                       aatype_mx=aatype_mx,
                       num_recycling=num_recycling)
    mx.eval(out)

    plddt_arr  = np.array(out['plddt'])[0, :L]
    coords     = np.array(out['final_atom_positions'])[0, :L]
    mask_out   = np.array(out['final_atom_mask'])[0, :L]
    mean_plddt = float(plddt_arr.mean())

    pdb_str = output_to_pdb(seq, coords, mask_out, plddt_arr)

    mx.metal.clear_cache()
    return pdb_str, mean_plddt, time.perf_counter() - t0


def _mlx_predict_batch(batch_items, num_recycling):
    """Run MLX MiniFold on a batch of sequences with the same recycling count.

    batch_items : list of (seq_id, seq)
    Returns list of (seq_id, pdb_str, mean_plddt) in the same order.
    """
    global _MLX_STATE
    import mlx.core as mx
    from minifold_mlx._data import prepare_input, pad_tokens, pad_mask, pad_aatype, output_to_pdb

    tokenizer, minifold_mlx, pad_id = _MLX_STATE

    prepared = [(seq_id, seq, *prepare_input(seq, tokenizer))
                for seq_id, seq in batch_items]
    # prepared entries: (seq_id, seq, tokens_mx, mask_np, aatype_np)

    max_len = max(p[2].shape[0] for p in prepared)

    tokens_mx = pad_tokens([p[2] for p in prepared], max_len, pad_id)
    mask_mx   = pad_mask(  [p[3] for p in prepared], max_len)
    aatype_mx = pad_aatype([p[4] for p in prepared], max_len)

    mx.eval(tokens_mx, mask_mx, aatype_mx)
    mx.metal.clear_cache()

    out = minifold_mlx(tokens_mx, seq_mask=mask_mx,
                       aatype_mx=aatype_mx,
                       num_recycling=num_recycling)
    mx.eval(out)

    plddt_all  = np.array(out['plddt'])
    coords_all = np.array(out['final_atom_positions'])
    mask_all   = np.array(out['final_atom_mask'])

    mx.metal.clear_cache()

    results = []
    for i, (seq_id, seq, tokens, _, _) in enumerate(prepared):
        L          = tokens.shape[0]
        plddt_arr  = plddt_all[i, :L]
        coords     = coords_all[i, :L]
        mask_out   = mask_all[i, :L]
        mean_plddt = float(plddt_arr.mean())
        results.append((seq_id, output_to_pdb(seq, coords, mask_out, plddt_arr), mean_plddt))

    return results


# ── PyTorch MiniFold fallback (Linux / WSL / Intel Mac) ──────────────────────

_PT_HF_CACHE = Path.home() / '.cache' / 'minifold'


def _load_pt_models(model_size='12L'):
    """Load PyTorch MiniFold on CUDA or CPU (non-Apple-Silicon fallback)."""
    global _PT_STATE
    if _PT_STATE is not None:
        return

    _write_status('Loading PyTorch MiniFold model...')
    print('  [PT] Loading PyTorch MiniFold...', flush=True)
    t0 = time.perf_counter()

    try:
        import minifold as _mf_pkg
    except ImportError:
        sys.exit(
            'ERROR: minifold (PyTorch) is not installed.\n'
            '  Run: pip install git+https://github.com/jwohlwend/minifold.git')

    # Locate predict.py shipped alongside the installed package
    import importlib.util
    pkg_root = os.path.dirname(os.path.dirname(_mf_pkg.__file__))
    for candidate in (
        os.path.join(pkg_root, 'predict.py'),
        os.path.join(os.path.dirname(_mf_pkg.__file__), 'predict.py'),
    ):
        if os.path.exists(candidate):
            predict_py = candidate
            break
    else:
        sys.exit(
            f'ERROR: predict.py not found near minifold package ({pkg_root}).\n'
            '  Re-install: pip install git+https://github.com/jwohlwend/minifold.git')

    spec = importlib.util.spec_from_file_location('minifold_predict', predict_py)
    predict_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_mod)

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _PT_HF_CACHE.mkdir(parents=True, exist_ok=True)
    predict_mod.download(_PT_HF_CACHE, model_size)
    checkpoint = _PT_HF_CACHE / f'minifold_{model_size}_final.ckpt'

    alphabet, model = predict_mod.create_model(checkpoint, device)
    model.eval()

    from minifold.model.config import model_config
    config = model_config('initial_training', train=False, low_prec=False,
                          long_sequence_inference=False)

    _PT_STATE = (predict_mod, alphabet, model, config, device)
    print(f'  [PT] Ready in {time.perf_counter()-t0:.1f}s  (device={device})',
          flush=True)


def _pt_predict(seq, num_recycling):
    """Run PyTorch MiniFold on a single sequence. Returns (pdb_str, mean_plddt, elapsed_s)."""
    global _PT_STATE
    import torch

    predict_mod, alphabet, model, config, device = _PT_STATE
    t0 = time.perf_counter()

    encoded_seq, mask, feats_of = predict_mod.prepare_input(seq, config, alphabet)
    model_batch = {
        'seq':      encoded_seq.unsqueeze(0).to(device),
        'mask':     mask.unsqueeze(0).to(device),
        'batch_of': {k: v.unsqueeze(0).to(device) for k, v in feats_of.items()},
    }

    use_amp = device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                 dtype=torch.bfloat16,
                                                 enabled=use_amp):
        out = model(model_batch, num_recycling=num_recycling)

    L          = len(seq)
    coords     = out['final_atom_positions'][0, :L].cpu().float().numpy()
    mask_out   = out['final_atom_mask'][0, :L].cpu().float().numpy()
    plddt_arr  = out['plddt'][0, :L].cpu().float().numpy()
    pdb_str    = predict_mod.output_to_pdb(seq, coords, mask_out, plddt_arr)
    return pdb_str, float(plddt_arr.mean()), time.perf_counter() - t0


def _pt_predict_batch(batch_items, num_recycling):
    """Run PyTorch MiniFold on a batch of (seq_id, seq) pairs.
    Returns list of (seq_id, pdb_str, mean_plddt).
    """
    global _PT_STATE
    import torch

    predict_mod, alphabet, model, config, device = _PT_STATE

    prepared = [(sid, seq, *predict_mod.prepare_input(seq, config, alphabet))
                for sid, seq in batch_items]

    max_len = max(p[2].shape[0] for p in prepared)

    def _pad1d(t, length, val=0):
        n = length - t.shape[0]
        return t if n == 0 else torch.cat([t, t.new_full((n,) + t.shape[1:], val)])

    seq_batch  = torch.stack([_pad1d(p[2], max_len, 1) for p in prepared]).to(device)
    mask_batch = torch.stack([_pad1d(p[3].long(), max_len, 0)
                              for p in prepared]).bool().to(device)
    of_keys  = list(prepared[0][4].keys())
    batch_of = {k: torch.stack([_pad1d(p[4][k], max_len, 0) for p in prepared]).to(device)
                for k in of_keys}

    model_batch = {'seq': seq_batch, 'mask': mask_batch, 'batch_of': batch_of}
    use_amp = device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                 dtype=torch.bfloat16,
                                                 enabled=use_amp):
        out = model(model_batch, num_recycling=num_recycling)

    results = []
    for i, (sid, seq, enc, _, _) in enumerate(prepared):
        L         = len(seq)
        coords    = out['final_atom_positions'][i, :L].cpu().float().numpy()
        mask_out  = out['final_atom_mask'][i, :L].cpu().float().numpy()
        plddt_arr = out['plddt'][i, :L].cpu().float().numpy()
        results.append((sid,
                        predict_mod.output_to_pdb(seq, coords, mask_out, plddt_arr),
                        float(plddt_arr.mean())))
    return results


# ── Backend dispatch ──────────────────────────────────────────────────────────

def _load_fold_models(model_size='12L', compile_miniformer=True, backend='auto'):
    """Load fold model for the active backend (MLX or PyTorch)."""
    global _USE_MLX
    if _USE_MLX is None:
        _USE_MLX = _resolve_backend(backend)
    if _USE_MLX:
        _load_mlx_models(model_size, compile_miniformer)
    else:
        _load_pt_models(model_size)


def _fold_predict(seq, num_recycling):
    """Predict structure for a single sequence (dispatches to MLX or PyTorch)."""
    with _GPU_LOCK:
        if _USE_MLX:
            return _mlx_predict(seq, num_recycling)
        return _pt_predict(seq, num_recycling)


def _fold_predict_batch(batch_items, num_recycling):
    """Predict structures for a batch of sequences (dispatches to MLX or PyTorch)."""
    with _GPU_LOCK:
        if _USE_MLX:
            return _mlx_predict_batch(batch_items, num_recycling)
        return _pt_predict_batch(batch_items, num_recycling)


# ── Missing-region fill (fold prediction + CCD + L-BFGS-B) ──────────────────

def _map_query_positions(domain_set, ops):
    qp     = 0
    result = []
    for op in ops:
        if op[0] == 'match':
            result.append((qp, op[1] in domain_set))
            qp += 1
        elif op[0] == 'insertion':
            for _ in op[1]:
                result.append((qp, False))
                qp += 1
    return result


def _find_missing_segments(qp_map, full_qseq, min_len):
    segs = []
    i    = 0
    while i < len(qp_map):
        qp_i, in_dom = qp_map[i]
        if not in_dom:
            j = i + 1
            while j < len(qp_map) and not qp_map[j][1]:
                j += 1
            end_qp = qp_map[j - 1][0] + 1
            subseq = full_qseq[qp_i:end_qp]
            prev_dom = qp_map[i - 1][0] if (i > 0 and qp_map[i - 1][1]) else None
            next_dom = qp_map[j][0]     if (j < len(qp_map) and qp_map[j][1]) else None
            if len(subseq) >= min_len:
                segs.append({
                    'qp_start': qp_i, 'qp_end': end_qp, 'seq': subseq,
                    'prev_domain_qp': prev_dom, 'next_domain_qp': next_dom,
                })
            i = j
        else:
            i += 1
    return segs


def _predict_fragments_inprocess(frag_dict, work_dir,
                                  plddt_threshold, max_recyclings,
                                  min_recycle_plddt=50.0,
                                  batch_tokens=262144, max_seq_len=800):
    """Predict fragments using batch multi-round recycling (same logic as Class D).
    Returns {name: pdb_path} for successes.
    """
    import traceback as _tb
    os.makedirs(work_dir, exist_ok=True)

    pending = []
    sec_pos_map = {}
    for name, seq in frag_dict.items():
        if max_seq_len > 0 and len(seq) > max_seq_len:
            print(f'    [{name}] SKIP: len={len(seq)} > max_seq_len={max_seq_len}',
                  flush=True)
            continue
        masked_seq, sec_pos = _mask_selenocysteine(seq)
        if sec_pos:
            sec_pos_map[name] = sec_pos
        pending.append((name, masked_seq))

    pending.sort(key=lambda x: len(x[1]))

    pdbs = {}
    still_pending = pending

    for r in range(max_recyclings + 1):
        if not still_pending:
            break

        batches, cur, cur_max = [], [], 0
        for item in still_pending:
            L = len(item[1])
            new_max = max(cur_max, L)
            if cur and (len(cur) + 1) * new_max * new_max > batch_tokens:
                batches.append(cur); cur, cur_max = [], 0; new_max = L
            cur.append(item); cur_max = new_max
        if cur:
            batches.append(cur)

        next_pending = []
        for batch in batches:
            try:
                batch_results = _fold_predict_batch([(n, seq) for n, seq in batch], r)
                for (name, pdb_str, mean_plddt), (_, seq) in zip(batch_results, batch):
                    floor_hit = (r == 0 and mean_plddt < min_recycle_plddt)
                    if mean_plddt >= plddt_threshold or r >= max_recyclings or floor_hit:
                        pdb_str = _restore_selenocysteine(pdb_str, sec_pos_map.get(name, []))
                        pdb_path = os.path.join(work_dir, f'{name}.pdb')
                        with open(pdb_path, 'w') as fh:
                            fh.write(pdb_str)
                        pdbs[name] = pdb_path
                        print(f'    [{name}] len={len(seq)} plddt={mean_plddt:.1f} r={r}',
                              flush=True)
                    else:
                        next_pending.append((name, seq))
            except Exception as exc:
                _tb.print_exc()
                print(f'    [batch] prediction failed: {exc}', flush=True)

        still_pending = next_pending

    return pdbs


def _frag_template_search(frag_dict: dict, work_dir: str) -> tuple[dict, dict]:
    """Search gap fragments against AFDB; return (still_novel, template_pdbs).

    still_novel : {name: seq} fragments with no good AFDB hit → go to MiniFold
    template_pdbs: {name: pdb_path} fragments covered by an AFDB template
    """
    import tempfile, subprocess as _sp
    if not (_SEARCH_BIN and _AFDB_SIDX_PATH and _AFDB_PQ_PATH and _FRAG_PDB_DIR):
        return frag_dict, {}
    if not frag_dict:
        return {}, {}

    # Write query FASTA
    fa_path = os.path.join(work_dir, '_frags_query.fa')
    with open(fa_path, 'w') as fh:
        for name, seq in frag_dict.items():
            fh.write(f'>{name}\n{seq}\n')

    # Run kmer search (top-3 hits per fragment, n_letters=5 same as pipeline)
    try:
        proc = _sp.run(
            [_SEARCH_BIN, _AFDB_SIDX_PATH, fa_path, '3', '0', '0', '9', '0', '5'],
            capture_output=True, text=True, timeout=30
        )
        lines = proc.stdout.strip().splitlines()
    except Exception:
        return frag_dict, {}

    # Parse: query  row_idx  shared  containment_value
    hits_by_frag: dict[str, list[tuple[int, float]]] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4 or parts[0] == 'query':
            continue
        name, row_idx, _, containment_value = parts[0], int(parts[1]), parts[2], float(parts[3])
        if containment_value >= _FRAG_MIN_CONTAINMENT:
            hits_by_frag.setdefault(name, []).append((row_idx, containment_value))

    if not hits_by_frag:
        return frag_dict, {}

    # Load AFDB parquet lazily (module-level cache to avoid repeated loads)
    global _afdb_pq_cache
    if _afdb_pq_cache is None:
        try:
            import polars as _pl
            _afdb_pq_cache = _pl.read_parquet(_AFDB_PQ_PATH)
        except Exception:
            return frag_dict, {}

    still_novel   = dict(frag_dict)
    template_pdbs = {}
    mat = parasail.blosum62

    for frag_name, hit_list in hits_by_frag.items():
        frag_seq = frag_dict.get(frag_name)
        if frag_seq is None:
            continue
        best_hit = None
        for row_idx, containment_value in sorted(hit_list, key=lambda x: -x[1]):
            try:
                db_row  = _afdb_pq_cache.row(row_idx, named=True)
                db_seq  = db_row['rep_AFDB_ID'], db_row['sequence']
                afdb_id, sseq = db_seq
            except Exception:
                continue
            r = parasail.sw_trace_striped_16(frag_seq, sseq, 10, 1, mat)
            qa, sa = r.traceback.query, r.traceback.ref
            pairs   = [(q, s) for q, s in zip(qa, sa) if q != '-' and s != '-']
            if not pairs:
                continue
            identity = sum(1 for q, s in pairs if q == s) / len(pairs)
            coverage = len(pairs) / len(frag_seq)
            if identity >= _FRAG_MIN_IDENTITY and coverage >= _FRAG_MIN_COVERAGE:
                best_hit = (afdb_id, sseq, qa, sa, identity, coverage)
                break

        if best_hit is None:
            continue

        afdb_id, sseq, qa, sa, identity, coverage = best_hit

        # Download AFDB PDB (uses existing local cache)
        local_pdb = _A.afdb_local_pdb(afdb_id, _FRAG_PDB_DIR)
        if not os.path.exists(local_pdb):
            try:
                from alphatracer.utils.afdb_fetch import AFDB_VERSION
                import urllib.request as _ur
                url = (f'https://alphafold.ebi.ac.uk/files/'
                       f'{afdb_id}-model_v{AFDB_VERSION}.pdb')
                _ur.urlretrieve(url, local_pdb)
            except Exception:
                continue

        # Extract fragment residues from the AFDB structure
        try:
            st     = gemmi.read_structure(local_pdb)
            chain  = st[0]['A']
            poly   = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]

            # Find start residue index in subject via SW alignment
            s_start = 0
            for c in sa:
                if c == '-':
                    s_start += 1
                else:
                    break

            frag_pdb_path = os.path.join(work_dir, f'{frag_name}_template.pdb')
            doc = gemmi.Document()
            new_st = gemmi.Structure()
            new_model = gemmi.Model('1')
            new_chain = gemmi.Chain('A')
            q_idx = 0
            s_idx = s_start
            for q_c, s_c in zip(qa, sa):
                if q_c != '-' and s_c != '-':
                    if s_idx < len(poly):
                        new_chain.add_residue(poly[s_idx])
                if s_c != '-':
                    s_idx += 1
            new_model.add_chain(new_chain)
            new_st.add_model(new_model)
            new_st.write_pdb(frag_pdb_path)

            still_novel.pop(frag_name, None)
            template_pdbs[frag_name] = frag_pdb_path
            print(f'    [{frag_name}] template from {afdb_id}  '
                  f'identity={identity*100:.1f}%  coverage={coverage*100:.1f}%',
                  flush=True)
        except Exception as exc:
            print(f'    [{frag_name}] template extraction failed: {exc}', flush=True)

    # ── ESM Atlas cascade: search fragments that had no good AFDB hit ────────
    if still_novel and _ESM_SIDX_PATH and _ESM_PQ_PATH and _FRAG_PDB_DIR:
        global _esm_pq_cache
        esm_fa = os.path.join(work_dir, '_frags_esm_query.fa')
        with open(esm_fa, 'w') as fh:
            for name, seq in still_novel.items():
                fh.write(f'>{name}\n{seq}\n')

        try:
            proc = _sp.run(
                [_SEARCH_BIN, _ESM_SIDX_PATH, esm_fa, '3', '0', '0', '9', '0', '5'],
                capture_output=True, text=True, timeout=30
            )
            esm_lines = proc.stdout.strip().splitlines()
        except Exception:
            esm_lines = []

        esm_hits_by_frag: dict[str, list[tuple[int, float]]] = {}
        for line in esm_lines:
            parts = line.split()
            if len(parts) < 4 or parts[0] == 'query':
                continue
            name, row_idx, containment_value = parts[0], int(parts[1]), float(parts[3])
            if containment_value >= _FRAG_MIN_CONTAINMENT:
                esm_hits_by_frag.setdefault(name, []).append((row_idx, containment_value))

        if esm_hits_by_frag:
            if _esm_pq_cache is None:
                try:
                    import polars as _pl
                    _esm_pq_cache = _pl.read_parquet(_ESM_PQ_PATH)
                except Exception:
                    esm_hits_by_frag = {}

        for frag_name, hit_list in esm_hits_by_frag.items():
            frag_seq = still_novel.get(frag_name)
            if frag_seq is None:
                continue
            best_hit = None
            for row_idx, containment_value in sorted(hit_list, key=lambda x: -x[1]):
                try:
                    db_row   = _esm_pq_cache.row(row_idx, named=True)
                    prot_hash = db_row['header']
                    esm_seq   = db_row['sequence']
                except Exception:
                    continue
                r = parasail.sw_trace_striped_16(frag_seq, esm_seq, 10, 1, mat)
                qa, sa = r.traceback.query, r.traceback.ref
                pairs  = [(q, s) for q, s in zip(qa, sa) if q != '-' and s != '-']
                if not pairs:
                    continue
                identity = sum(1 for q, s in pairs if q == s) / len(pairs)
                coverage = len(pairs) / len(frag_seq)
                if identity >= _FRAG_MIN_IDENTITY and coverage >= _FRAG_MIN_COVERAGE:
                    best_hit = (prot_hash, esm_seq, qa, sa, identity, coverage)
                    break

            if best_hit is None:
                continue

            prot_hash, esm_seq, qa, sa, identity, coverage = best_hit

            # Fetch ESM PDB if not cached
            from alphatracer.utils.esm_atlas_fetch import (
                fetch_esm_structures, esm_local_pdb as _esm_local_pdb)
            local_pdb = _esm_local_pdb(prot_hash, _FRAG_PDB_DIR)
            if not os.path.exists(local_pdb):
                try:
                    fetch_esm_structures(
                        [{'protein_hash': prot_hash}], _FRAG_PDB_DIR)
                except Exception as e:
                    print(f'    [{frag_name}] ESM fetch failed: {e}', flush=True)
                    continue

            if not os.path.exists(local_pdb):
                continue

            # Extract fragment residues from ESM structure
            try:
                st    = gemmi.read_structure(local_pdb)
                chain = st[0]['A']
                poly  = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]

                s_start = 0
                for c in sa:
                    if c == '-':
                        s_start += 1
                    else:
                        break

                frag_pdb_path = os.path.join(work_dir, f'{frag_name}_template.pdb')
                new_st    = gemmi.Structure()
                new_model = gemmi.Model('1')
                new_chain = gemmi.Chain('A')
                s_idx = s_start
                for q_c, s_c in zip(qa, sa):
                    if q_c != '-' and s_c != '-':
                        if s_idx < len(poly):
                            new_chain.add_residue(poly[s_idx])
                    if s_c != '-':
                        s_idx += 1
                new_model.add_chain(new_chain)
                new_st.add_model(new_model)
                new_st.write_pdb(frag_pdb_path)

                still_novel.pop(frag_name, None)
                template_pdbs[frag_name] = frag_pdb_path
                print(f'    [{frag_name}] ESM template from {prot_hash[:16]}  '
                      f'identity={identity*100:.1f}%  coverage={coverage*100:.1f}%',
                      flush=True)
            except Exception as exc:
                print(f'    [{frag_name}] ESM template extraction failed: {exc}', flush=True)

    return still_novel, template_pdbs


def _read_backbone_residues(pdb_path):
    st = gemmi.read_structure(pdb_path)
    result = []
    for model in st:
        for chain in model:
            for res in chain:
                atoms = {}
                for aname in ('N', 'CA', 'C', 'O', 'CB'):
                    a = res.find_atom(aname, '\0')
                    if a:
                        atoms[aname] = np.array([a.pos.x, a.pos.y, a.pos.z])
                if 'CA' in atoms:
                    result.append({'resname': res.name, 'atoms': atoms,
                                   'from_ref': False})
        break
    return result


def _apply_rigid(residues, R, t):
    return [
        {**r, 'atoms': {k: v @ R.T + t for k, v in r['atoms'].items()}}
        for r in residues
    ]


def _place_fragment_lbfgsb(frag_residues, pre_ca, post_ca, n_iters=300):
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as Rot

    ca_list = [r['atoms']['CA'] for r in frag_residues if 'CA' in r['atoms']]
    if not ca_list:
        return frag_residues

    ca_arr   = np.array(ca_list)
    centroid = ca_arr.mean(0)
    ca_c     = ca_arr - centroid

    def _get_ca(params):
        t  = params[:3]
        rv = params[3:6]
        R  = (Rot.from_rotvec(rv).as_matrix()
              if np.linalg.norm(rv) > 1e-8 else np.eye(3))
        return ca_c @ R.T + centroid + t

    def obj(params):
        ca_t = _get_ca(params)
        loss = 0.0
        if pre_ca is not None:
            loss += (np.linalg.norm(ca_t[0]  - pre_ca)  - 3.8) ** 2
        if post_ca is not None:
            loss += (np.linalg.norm(ca_t[-1] - post_ca) - 3.8) ** 2
        return loss

    x0 = np.zeros(6)
    if pre_ca is not None:
        x0[:3] = pre_ca - ca_arr[0]
    elif post_ca is not None:
        x0[:3] = post_ca - ca_arr[-1]

    res = minimize(obj, x0, method='L-BFGS-B',
                   options={'maxiter': n_iters, 'ftol': 1e-12, 'gtol': 1e-8})

    t_opt = res.x[:3]
    rv    = res.x[3:6]
    R_opt = (Rot.from_rotvec(rv).as_matrix()
             if np.linalg.norm(rv) > 1e-8 else np.eye(3))
    return _apply_rigid(frag_residues, R_opt, t_opt)


def build_complete_structure(
    domain_indices, ops, ref_poly, full_qseq, out_pdb, work_dir,
    mm_iters=300, ccd_iters=200, ccd_tol=0.15, n_flank=3,
    min_frag_len=5, lbfgsb_iters=300, model_size='12L', anchor_k=1000.0,
    plddt_threshold=85.0, max_recyclings=2, min_recycle_plddt=50.0,
    batch_tokens=262144, max_seq_len=800, loop_closer='ccd',
):
    t0 = time.perf_counter()
    try:
        domain_set = set(domain_indices)
        d_ops      = _domain_ops(domain_set, ops)
        ref_ss     = _B.compute_ref_ss(ref_poly)

        indel_ctx = {}
        for k, op in enumerate(d_ops):
            if op[0] in ('deletion', 'insertion'):
                indel_ctx[k] = _B._indel_ss_context(k, d_ops, ref_ss)

        d_op_id_to_didx = {id(op): i for i, op in enumerate(d_ops)}

        def _ref_backbone(res):
            atoms = {}
            for aname in ('N', 'CA', 'C', 'O', 'CB'):
                a = res.find_atom(aname, '\0')
                if a:
                    atoms[aname] = np.array([a.pos.x, a.pos.y, a.pos.z])
            return atoms

        domain_seg     = []
        dom_qp_to_ridx = {}

        qp = 0
        for op in ops:
            if op[0] == 'match':
                _, rpos, qaa = op
                if rpos in domain_set:
                    if rpos >= len(ref_poly):
                        return False, f'ref_pos {rpos} OOB', 0.
                    resname = ONE_TO_THREE.get(qaa, 'ALA')
                    atoms   = _ref_backbone(ref_poly[rpos])
                    keep    = {'N', 'CA', 'C', 'O'} | ({'CB'} if qaa != 'G' else set())
                    atoms   = {k: v for k, v in atoms.items() if k in keep}
                    if len({'N', 'CA', 'C', 'O'} & atoms.keys()) < 4:
                        return False, f'incomplete backbone ref {rpos}', 0.
                    dom_qp_to_ridx[qp] = len(domain_seg)
                    domain_seg.append({'resname': resname, 'atoms': atoms,
                                       'from_ref': True, 'qp': qp,
                                       'ref_pos': rpos})
                qp += 1

            elif op[0] == 'deletion':
                pass

            elif op[0] == 'insertion':
                didx   = d_op_id_to_didx.get(id(op))
                in_dom = didx is not None
                for aa in op[1]:
                    if in_dom and domain_seg:
                        prev = domain_seg[-1]['atoms']
                        if all(k in prev for k in ('N', 'CA', 'C')):
                            pN, pCA, pC = prev['N'], prev['CA'], prev['C']
                            ctx     = indel_ctx.get(didx, 'C')
                            builder = (_B.build_helix_residue if ctx == 'H'
                                       else _B.build_extended_residue)
                            resname   = ONE_TO_THREE.get(aa, 'ALA')
                            new_atoms = builder(pN, pCA, pC, resname)
                            dom_qp_to_ridx[qp] = len(domain_seg)
                            domain_seg.append({'resname': resname,
                                               'atoms': new_atoms,
                                               'from_ref': False, 'qp': qp})
                    qp += 1

        if not domain_seg:
            return False, 'no domain residues built', 0.

        qp_map  = _map_query_positions(domain_set, ops)
        missing = _find_missing_segments(qp_map, full_qseq, min_frag_len)

        if not missing:
            return build_domain_pdb(
                domain_indices, ops, ref_poly, out_pdb,
                mm_iters, ccd_iters, ccd_tol, n_flank, anchor_k=anchor_k,
                loop_closer=loop_closer)

        frag_dict = {f'frag_{i}': seg['seq'] for i, seg in enumerate(missing)}
        # Search fragments against AFDB before falling back to MiniFold
        novel_frags, template_pdbs = _frag_template_search(frag_dict, work_dir)
        try:
            frag_pdbs = _predict_fragments_inprocess(
                novel_frags, work_dir,
                plddt_threshold=plddt_threshold,
                max_recyclings=max_recyclings,
                min_recycle_plddt=min_recycle_plddt,
                batch_tokens=batch_tokens,
                max_seq_len=max_seq_len,
            )
        except Exception as exc:
            print(f'  [complete] fragment prediction error — '
                  f'falling back to domain-only: {exc}')
            return build_domain_pdb(
                domain_indices, ops, ref_poly, out_pdb,
                mm_iters, ccd_iters, ccd_tol, n_flank, anchor_k=anchor_k)

        frag_pdbs.update(template_pdbs)

        all_segs = []
        dom_qp_start = min(r['qp'] for r in domain_seg)
        dom_qp_end   = max(r['qp'] for r in domain_seg) + 1
        all_segs.append((dom_qp_start, dom_qp_end, domain_seg))

        for i, seg in enumerate(missing):
            frag_id = f'frag_{i}'
            if frag_id not in frag_pdbs:
                continue
            frag_res = _read_backbone_residues(frag_pdbs[frag_id])
            if not frag_res:
                continue

            pre_ca = post_ca = None
            prev_qp = seg['prev_domain_qp']
            next_qp = seg['next_domain_qp']
            if prev_qp is not None and prev_qp in dom_qp_to_ridx:
                pre_ca = domain_seg[dom_qp_to_ridx[prev_qp]]['atoms'].get('CA')
            if next_qp is not None and next_qp in dom_qp_to_ridx:
                post_ca = domain_seg[dom_qp_to_ridx[next_qp]]['atoms'].get('CA')

            frag_res = _place_fragment_lbfgsb(frag_res, pre_ca, post_ca, lbfgsb_iters)
            all_segs.append((seg['qp_start'], seg['qp_end'], frag_res))

        all_segs.sort(key=lambda x: x[0])

        residues  = []
        gap_sites = []
        for s_idx, (_, _, seg_res) in enumerate(all_segs):
            if s_idx > 0 and residues:
                gap_sites.append(len(residues) - 1)
            residues.extend(seg_res)

        if not residues:
            return False, 'no residues after assembly', 0.

        if gap_sites:
            flat_atoms  = []
            flat_pos    = []
            atom_to_idx = {}
            for ri, res in enumerate(residues):
                for aname in ('N', 'CA', 'C', 'O', 'CB'):
                    if aname in res['atoms']:
                        atom_to_idx[(ri, aname)] = len(flat_atoms)
                        flat_atoms.append((ri, aname))
                        flat_pos.append(res['atoms'][aname].copy())

            for gap_before_ri in gap_sites:
                after_ri   = gap_before_ri + 1
                if after_ri >= len(residues):
                    continue
                target_idx = atom_to_idx.get((after_ri, 'N'))
                if target_idx is None:
                    continue
                target  = flat_pos[target_idx].copy()
                end_idx = atom_to_idx.get((gap_before_ri, 'C'))
                if end_idx is None:
                    continue
                m_start = max(0, gap_before_ri + 1 - n_flank)
                m_end   = gap_before_ri + 1
                pivot_bonds = []
                for ri in range(m_start, m_end):
                    for a_name, b_name in (('N', 'CA'), ('CA', 'C')):
                        ia = atom_to_idx.get((ri, a_name))
                        ib = atom_to_idx.get((ri, b_name))
                        if ia is None or ib is None:
                            continue
                        downstream = [
                            atom_to_idx[(rj, an)]
                            for rj in range(ri, m_end)
                            for an in ('N', 'CA', 'C', 'O', 'CB')
                            if (rj, an) in atom_to_idx and atom_to_idx[(rj, an)] > ib
                        ]
                        if end_idx > ib and end_idx not in downstream:
                            downstream.append(end_idx)
                        if downstream:
                            pivot_bonds.append((ia, ib, downstream))
                if pivot_bonds:
                    _B.ccd_close(flat_pos, pivot_bonds, end_idx, target,
                                 max_iter=ccd_iters, tol=ccd_tol)

            for (ri, aname), pos in zip(flat_atoms, flat_pos):
                residues[ri]['atoms'][aname] = pos

        anchors = _find_segment_anchors(residues, ref_poly)
        system, top, pos_nm = _B.build_openmm_system(residues)
        _add_anchor_restraints(system, residues, anchors, anchor_k)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
        )
        sim = Simulation(top, system, integrator, platform=_B._get_platform())
        sim.context.setPositions(pos_nm)
        sim.minimizeEnergy(maxIterations=mm_iters)
        state = sim.context.getState(getPositions=True)
        with open(out_pdb, 'w') as fh:
            PDBFile.writeFile(top, state.getPositions(), fh)

        return True, None, time.perf_counter() - t0

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return False, str(exc), time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args           = parse_args()
    global _STATUS_PATH
    indir          = args.input_dir.rstrip('/')
    pdb_dir        = os.path.join(indir, 'AF_pdbs')
    pae_dir        = os.path.join(indir, 'AF_pae')
    outdir_C       = os.path.join(indir, 'output_pdbs_classC')
    outdir_D       = os.path.join(indir, 'output_pdbs_classD')
    threshold_frac = args.min_pctsim / 100.0
    _STATUS_PATH   = os.path.join(indir, '.cd_status')

    os.makedirs(outdir_C, exist_ok=True)
    os.makedirs(pae_dir,  exist_ok=True)
    os.makedirs(pdb_dir,  exist_ok=True)

    print('=' * 60)
    print('AlphaTracer 1.0  —  Class C + D Pipeline')
    print('=' * 60)
    print(f'  Input dir:      {indir}/')
    print(f'  Min pctsim:     {args.min_pctsim}%  (window={args.window_size} aa)')
    _step_label = 'auto (2/3)' if args.pae_step == 0 else str(args.pae_step)
    print(f'  PAE cutoff:     {args.pae_cutoff} Å  '
          f'power={args.pae_power}  resolution={args.pae_resolution}  '
          f'step={_step_label}  (>{args.large_domain_threshold} aa: {args.pae_resolution_large})')
    print(f'  CCD:            {args.ccd_iters} iters, tol={args.ccd_tol} Å, '
          f'flank={args.flank}')
    print(f'  OpenMM:         {args.mm_iters} minimisation steps')
    print(f'  Anchor k:       {args.anchor_k} kJ/mol/nm²')
    if args.fill_missing:
        print(f'  Fill missing:   ON  (MLX MiniFold {args.mini_model_size}, '
              f'min_frag={args.min_frag_len} aa, L-BFGS-B={args.lbfgsb_iters})')
    print(f'  Recycling:      pLDDT-gated (threshold={args.plddt_threshold}, '
          f'max={args.max_recyclings}, floor={args.min_recycle_plddt})')
    print(f'  MLX compile:    {"OFF (--no-compile)" if args.no_compile else "ON (mx.compile MiniFormer)"}')
    print(f'  Max seq len:    {args.max_seq_len if args.max_seq_len > 0 else "unlimited"}'
          f'  (MLX Metal OOM guard)')
    print(f'  Class D:        {"SKIP" if args.no_classD else "ON"}'
          + (f'  (limit={args.classD_limit})' if args.classD_limit > 0 else ''))
    print()

    # ── Load hits ─────────────────────────────────────────────────────────────
    hits_df = None
    for fname in ['allhits.pq', 'tophits.pq', 'kmer_hits.pq']:
        p = os.path.join(indir, fname)
        if os.path.exists(p):
            hits_df = pl.read_parquet(p)
            print(f'Loaded {fname}: {len(hits_df)} rows')
            break
    if hits_df is None:
        sys.exit(f'No allhits.pq/tophits.pq/kmer_hits.pq in {indir}')

    classA_path = os.path.join(indir, 'classA.pq')
    if not os.path.exists(classA_path):
        sys.exit(f'classA.pq not found in {indir}')
    classA_ids = set(pl.read_parquet(classA_path)['qseqid'].to_list())

    classB_path = os.path.join(indir, 'classB.pq')
    classB_ids  = (set(pl.read_parquet(classB_path)['qseqid'].to_list())
                   if os.path.exists(classB_path) else set())

    # ── [C-1] Alignment ───────────────────────────────────────────────────────
    exclude_ids = classA_ids | classB_ids
    _sort_col = 'containment_value' if 'containment_value' in hits_df.columns else 'approx_pident'
    candidates = hits_df
    if 'approx_pident' in hits_df.columns:
        candidates = candidates.filter(pl.col('approx_pident') >= args.min_pctsim)
    top_k = args.classC_top_k
    candidates = (
        candidates
        .filter(~pl.col('qseqid').is_in(exclude_ids))
        .sort(_sort_col, descending=True)
        .group_by('qseqid', maintain_order=True)
        .agg(pl.all().head(top_k))
        .explode(pl.all().exclude('qseqid'))
    )
    n_queries = candidates['qseqid'].n_unique()
    print(f'{len(classA_ids)} Class A + {len(classB_ids)} Class B excluded  |  '
          f'{n_queries} queries for Class C ({len(candidates)} candidates, top_k={top_k})')
    with open(os.path.join(indir, '.classC_total'), 'w') as _f:
        _f.write(str(n_queries))

    if args.limit > 0:
        keep_ids = candidates['qseqid'].unique()[:args.limit].to_list()
        candidates = candidates.filter(pl.col('qseqid').is_in(keep_ids))
        print(f'  (limited to first {args.limit} queries)')

    _write_status(f'Aligning {len(candidates)} Class C candidates ({n_queries} queries)...')
    print(f'\n[C-1/4] Aligning {len(candidates)} candidates for {n_queries} queries '
          f'(threads={args.threads})...')
    rows_list = list(candidates.iter_rows(named=True))
    total     = len(rows_list)

    def _aln_c(row):
        return _B.align_nw(row['qseqid'], row['sseqid'],
                           row['full_qseq'], row['full_sseq'])

    aln_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for result in ex.map(_aln_c, rows_list):
            done += 1
            if result:
                aln_rows.append(result)
            if done % 200 == 0 or done == total:
                print(f'  {done}/{total}...', end='\r')
    print()

    if not aln_rows:
        print('No alignments.'); return

    aln_df = pl.DataFrame(aln_rows,
                          schema=['qseqid', 'sseqid', 'qseq_alg',
                                  'sseq_alg', 'alg_comp'],
                          orient='row')
    # Select best hit per query by alignment coverage (matched residues / qlen)
    def _coverage(r):
        return sum(1 for a, b in zip(r['qseq_alg'], r['sseq_alg'])
                   if a != '-' and b != '-') / max(r['qlen'], 1)

    merged = (
        candidates.join(aln_df, on=['qseqid', 'sseqid'], how='inner')
        .with_columns(
            pl.struct(['qseq_alg', 'sseq_alg', 'qlen'])
            .map_elements(_coverage, return_dtype=pl.Float64)
            .alias('aln_coverage')
        )
        .sort('aln_coverage', descending=True)
        .group_by('qseqid', maintain_order=True)
        .agg(pl.all().first())
        .drop('aln_coverage')
    )

    # ── [C-1b] C1/C2 split: classify by alignment coverage + identity ─────────
    # C1: NW coverage ≥ 70% of qlen AND identity ≥ min_pctsim → use full aligned
    #     region as template; MiniFold fills only the unmatched gaps.
    # C2: weaker hits → PAE-based compact domain detection (original behaviour).
    C1_COV_THRESH = 0.70
    c1_rows_raw, c2_rows_raw = [], []
    for row in merged.iter_rows(named=True):
        qa, sa = row['qseq_alg'], row['sseq_alg']
        qlen   = len(row['full_qseq'])
        pairs  = [(q, s) for q, s in zip(qa, sa) if q != '-' and s != '-']
        if not pairs:
            c2_rows_raw.append(row); continue
        identity = sum(1 for q, s in pairs if q == s) / len(pairs)
        coverage = len(pairs) / qlen
        if coverage >= C1_COV_THRESH and identity >= threshold_frac:
            c1_rows_raw.append(row)
        else:
            c2_rows_raw.append(row)
    print(f'  C1 (high-coverage template): {len(c1_rows_raw)}  '
          f'C2 (PAE domain):              {len(c2_rows_raw)}')

    # ── [C-2+3/4-C1] C1: Download PDB only; domain = full aligned region ──────
    c1_afdb_ids = set()
    c1_esm_rows = []
    for row in c1_rows_raw:
        aid = _A.get_afdb_id(row['sseqid'])
        if aid:
            c1_afdb_ids.add(aid)
        else:
            c1_esm_rows.append(row)

    _write_status(f'Downloading PDB files for {len(c1_afdb_ids)} C1 templates...')
    print(f'\n[C-2/4-C1] Downloading {len(c1_afdb_ids)} C1 PDB files...')
    _B.stage_download(c1_afdb_ids, pdb_dir, args.threads)
    _ref_poly_cache = {}

    def _load_ref_poly(afdb_id):
        src_pdb = _A.afdb_local_pdb(afdb_id, pdb_dir)
        if not os.path.exists(src_pdb):
            return afdb_id, None
        try:
            st   = gemmi.read_structure(src_pdb)
            poly = [r for r in st[0]['A']
                    if r.entity_type == gemmi.EntityType.Polymer]
            return afdb_id, poly
        except Exception:
            return afdb_id, None

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for aid, poly in ex.map(_load_ref_poly, sorted(c1_afdb_ids)):
            if poly is not None:
                _ref_poly_cache[aid] = poly

    # ESM PDBs for C1
    if c1_esm_rows:
        _resolve_esm_fragment_ids(c1_esm_rows)
        fetch_esm_structures(
            [{'protein_hash': r.get('protein_hash') or r['sseqid'].split('|')[0],
              'fragment_id':  r.get('fragment_id', -1),
              'frag_row':     r.get('frag_row', -1)}
             for r in c1_esm_rows],
            pdb_dir, n_workers=args.threads, pae_dir=pae_dir,
        )
        seen_ph = set()
        for row in c1_esm_rows:
            ph = row.get('protein_hash') or row['sseqid'].split('|')[0]
            if ph in seen_ph or ph in _ref_poly_cache:
                continue
            seen_ph.add(ph)
            src_pdb = esm_local_pdb(ph, pdb_dir)
            if not os.path.exists(src_pdb):
                continue
            try:
                st   = gemmi.read_structure(src_pdb)
                poly = list(st[0][0])
                _ref_poly_cache[ph] = poly
            except Exception:
                pass

    def _qualify_c1(row):
        """C1: domain = all matched reference positions from NW alignment."""
        aid  = _A.get_afdb_id(row['sseqid'])
        key  = aid if aid else (row.get('protein_hash') or row['sseqid'].split('|')[0])
        ref_poly = _ref_poly_cache.get(key)
        if ref_poly is None:
            return None, 'no_pdb'
        _, ops = _B.parse_alignment_ops(row['qseq_alg'], row['sseq_alg'])
        domain = [op[1] for op in ops if op[0] == 'match']
        if len(domain) < args.min_domain_size:
            return None, 'domain_too_small'
        plddt_vals = [ref_poly[pos].find_atom('CA', '\0').b_iso
                      for pos in domain if pos < len(ref_poly)
                      and ref_poly[pos].find_atom('CA', '\0')]
        if plddt_vals and (sum(plddt_vals) / len(plddt_vals)) < args.min_domain_plddt:
            return None, 'domain_low_plddt'
        return {**row, '_best_domain': domain, '_c1': True}, 'ok'

    c1_qual_results = [_qualify_c1(row) for row in c1_rows_raw]

    # ── [C-2+3/4-C2] C2: PAE download + domain detection (original logic) ──────
    c2_afdb_ids_map = {}
    c2_esm_rows_list = []
    for row in c2_rows_raw:
        aid = _A.get_afdb_id(row['sseqid'])
        if aid:
            c2_afdb_ids_map.setdefault(aid, []).append(row)
        else:
            c2_esm_rows_list.append(row)

    all_c2_afdb_ids = set(c2_afdb_ids_map.keys())

    _write_status(f'Downloading PAE/PDB and detecting domains for {len(c2_rows_raw)} C2 sequences...')
    print(f'\n[C-2+3/4-C2] Downloading PAE+PDB and qualifying domains '
          f'({len(c2_rows_raw)} sequences, threads={args.threads})...')

    # Phase 2a: PAE
    print('  [2a] Downloading PAE files (C2 only)...')
    fetch_afdb_pae(all_c2_afdb_ids, pae_dir)

    # Phase 2b: PDB
    print('  [2b] Downloading PDB files (C2)...')
    _B.stage_download(all_c2_afdb_ids, pdb_dir, args.threads)

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for aid, poly in ex.map(_load_ref_poly, sorted(all_c2_afdb_ids)):
            if poly is not None:
                _ref_poly_cache[aid] = poly

    if c2_esm_rows_list:
        _resolve_esm_fragment_ids(c2_esm_rows_list)
        print(f'  [2b-ESM] Fetching {len(c2_esm_rows_list)} ESM Atlas structure(s)...')
        fetch_esm_structures(
            [{'protein_hash': r.get('protein_hash') or r['sseqid'].split('|')[0],
              'fragment_id':  r.get('fragment_id', -1),
              'frag_row':     r.get('frag_row', -1)}
             for r in c2_esm_rows_list],
            pdb_dir, n_workers=args.threads, pae_dir=pae_dir,
        )
        seen_ph = set()
        for row in c2_esm_rows_list:
            ph = row.get('protein_hash') or row['sseqid'].split('|')[0]
            if ph in seen_ph or ph in _ref_poly_cache:
                continue
            seen_ph.add(ph)
            src_pdb = esm_local_pdb(ph, pdb_dir)
            if not os.path.exists(src_pdb):
                continue
            try:
                st   = gemmi.read_structure(src_pdb)
                poly = list(st[0][0])
                _ref_poly_cache[ph] = poly
            except Exception:
                pass

    # Phase 2c: Batch Rust PAE parse
    print('  [2c] Batch-parsing PAE matrices (Rust)...')
    _t2c = time.time()
    _SIZE_THRESH = 350_000
    pae_paths_map = {aid: _pae_local_path(aid, pae_dir)
                     for aid in all_c2_afdb_ids
                     if os.path.exists(_pae_local_path(aid, pae_dir))}
    pae_matrices_by_afdb = {}
    if pae_paths_map:
        if args.pae_step:
            groups = [(args.pae_step, pae_paths_map)]
        else:
            small = {a: p for a, p in pae_paths_map.items() if os.path.getsize(p) < _SIZE_THRESH}
            large = {a: p for a, p in pae_paths_map.items() if os.path.getsize(p) >= _SIZE_THRESH}
            groups = [(2, small), (3, large)]
        for step, group in groups:
            if not group:
                continue
            try:
                aid_list  = list(group.keys())
                path_list = [group[a] for a in aid_list]
                matrices  = parse_pae_batch_rust(path_list, step=step)
                for aid, path in zip(aid_list, path_list):
                    if path in matrices:
                        n_full, mat = matrices[path]
                        pae_matrices_by_afdb[aid] = (n_full, mat, step)
            except Exception as e:
                print(f'  WARNING: Rust batch PAE parse (step={step}) failed ({e}); falling back to Python')
                for aid, path in group.items():
                    try:
                        pae = _load_pae_matrix(path)
                        pae_matrices_by_afdb[aid] = (pae.shape[0], pae.astype('float32'), 1)
                    except Exception:
                        pass

    n_esm_pae = 0
    seen_esm_ph = set()
    for row in c2_esm_rows_list:
        ph = row.get('protein_hash') or row['sseqid'].split('|')[0]
        if ph in seen_esm_ph or ph in pae_matrices_by_afdb:
            continue
        seen_esm_ph.add(ph)
        npy_path = os.path.join(pae_dir, f'esm_{ph}.pae.npy')
        if os.path.exists(npy_path):
            try:
                mat = np.load(npy_path).astype(np.float32)
                pae_matrices_by_afdb[ph] = (mat.shape[0], mat, 1)
                n_esm_pae += 1
            except Exception:
                pass
    if c2_esm_rows_list:
        print(f'  ESM PAE loaded: {n_esm_pae}/{len(seen_esm_ph)} matrices')
    print(f'  PAE parse: {time.time()-_t2c:.2f}s  ({len(pae_matrices_by_afdb)} matrices)')
    _t2d = time.time()

    # Phase 3: domain detection for C2
    def _qualify_esm_row(row):
        ph = row.get('protein_hash') or row['sseqid'].split('|')[0]
        ref_poly = _ref_poly_cache.get(ph)
        if ref_poly is None:
            return None, 'no_pdb'
        domain = list(range(len(ref_poly)))
        if len(domain) < args.min_domain_size:
            return None, 'domain_too_small'
        _, ops     = _B.parse_alignment_ops(row['qseq_alg'], row['sseq_alg'])
        rp_to_comp = _ref_pos_to_comp(row['qseq_alg'], row['sseq_alg'], row['alg_comp'])
        domain_set = set(domain)
        d_ops      = _domain_ops(domain_set, ops)
        if not d_ops:
            return None, 'no_domain'
        comp = _domain_comp_str(domain_set, d_ops, rp_to_comp)
        if not _A.all_windows_pass(comp, args.window_size, threshold_frac):
            return None, 'no_domain'
        return {**row, '_best_domain': domain}, 'ok'

    def _qualify_row(row, ref_poly, n_full, pae_matrix, step):
        _, ops     = _B.parse_alignment_ops(row['qseq_alg'], row['sseq_alg'])
        rp_to_comp = _ref_pos_to_comp(row['qseq_alg'], row['sseq_alg'], row['alg_comp'])
        ref_ss     = _B.compute_ref_ss(ref_poly)
        pae_res    = (args.pae_resolution_large
                      if n_full > args.large_domain_threshold
                      else args.pae_resolution)

        # Try multiple PAE cutoffs and pick the domain maximising size × identity.
        # Tighter cutoffs (smaller pae_cutoff) give more, smaller domains;
        # looser cutoffs merge them. Trying both recovers large coherent domains
        # that a single tight cutoff would fragment.
        _PAE_CUTOFFS = sorted(set([args.pae_cutoff, 8.0, 12.0]))
        all_qualifying = []
        for cutoff in _PAE_CUTOFFS:
            domains = get_domains_from_matrix(pae_matrix, n_full, step,
                                              args.pae_power, cutoff, pae_res)
            for d in domains:
                if classify_domain(d, ops, ref_ss, rp_to_comp,
                                   args.window_size, threshold_frac):
                    matches = sum(1 for r in d if rp_to_comp.get(r) == '|')
                    score   = len(d) * (matches / len(d) if d else 0)
                    all_qualifying.append((score, d))

        if not all_qualifying:
            return None, 'no_domain'
        best = max(all_qualifying, key=lambda x: x[0])[1]
        if len(best) < args.min_domain_size:
            return None, 'domain_too_small'
        plddt_vals = [ref_poly[pos].find_atom('CA', '\0').b_iso
                      for pos in best if pos < len(ref_poly)
                      and ref_poly[pos].find_atom('CA', '\0')]
        if plddt_vals and (sum(plddt_vals) / len(plddt_vals)) < args.min_domain_plddt:
            return None, 'domain_low_plddt'
        return {**row, '_best_domain': best}, 'ok'

    def _qualify_with_sw_fallback(row, ref_poly, n_full, pae_matrix, step):
        """Retry C2 qualification with SW alignment when NW gives no_domain."""
        try:
            r      = parasail.sw_trace_striped_16(
                row['full_qseq'], row['full_sseq'], 10, 1, parasail.blosum45)
            sw_qa  = r.traceback.query
            sw_sa  = r.traceback.ref
            sw_row = {**row, 'qseq_alg': sw_qa, 'sseq_alg': sw_sa,
                      'alg_comp': r.traceback.comp.replace(' ', '-')}
        except Exception:
            return None, 'no_domain'
        # Check C1 promotion first
        pairs = [(q, s) for q, s in zip(sw_qa, sw_sa) if q != '-' and s != '-']
        if pairs:
            identity = sum(1 for q, s in pairs if q == s) / len(pairs)
            coverage = len(pairs) / len(row['full_qseq'])
            if coverage >= C1_COV_THRESH and identity >= threshold_frac:
                result, status = _qualify_c1(sw_row)
                if status == 'ok':
                    result['_sw_rescue'] = True
                return result, status
        result, status = _qualify_row(sw_row, ref_poly, n_full, pae_matrix, step)
        if status == 'ok' and result:
            result['_sw_rescue'] = True
        return result, status

    def _qualify_afdb(afdb_id, rows):
        if afdb_id is None:
            return [(None, 'skip')] * len(rows)
        ref_poly = _ref_poly_cache.get(afdb_id)
        if ref_poly is None:
            return [(None, 'skip')] * len(rows)
        pae_data = pae_matrices_by_afdb.get(afdb_id)
        if pae_data is None:
            return [(None, 'no_pae')] * len(rows)
        n_full, pae_matrix, step = pae_data
        results = []
        for row in rows:
            result, status = _qualify_row(row, ref_poly, n_full, pae_matrix, step)
            if status == 'no_domain':
                result, status = _qualify_with_sw_fallback(
                    row, ref_poly, n_full, pae_matrix, step)
            results.append((result, status))
        return results

    c2_qual_results = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(_qualify_afdb, aid, rows): aid
                   for aid, rows in c2_afdb_ids_map.items()}
        for fut in as_completed(futures):
            c2_qual_results.extend(fut.result())

    def _qualify_esm_dispatch(row):
        ph       = row.get('protein_hash') or row['sseqid'].split('|')[0]
        pae_data = pae_matrices_by_afdb.get(ph)
        if pae_data is not None:
            ref_poly = _ref_poly_cache.get(ph)
            if ref_poly is None:
                return None, 'no_pdb'
            n_full, pae_matrix, step = pae_data
            result, status = _qualify_row(row, ref_poly, n_full, pae_matrix, step)
            if status == 'no_domain':
                result, status = _qualify_with_sw_fallback(
                    row, ref_poly, n_full, pae_matrix, step)
            return result, status
        return _qualify_esm_row(row)

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for result in ex.map(_qualify_esm_dispatch, c2_esm_rows_list):
            c2_qual_results.append(result)

    print(f'  Domain inference: {time.time()-_t2d:.2f}s  ({len(c2_qual_results)} C2 candidates)')

    # ── Merge C1 + C2 qualifying results ──────────────────────────────────────
    qual_results = c1_qual_results + c2_qual_results

    classC_rows = []
    n_qualify = n_no_pae = n_no_domain = n_too_small = n_low_plddt = n_no_pdb = 0
    n_c1_ok = n_c2_ok = n_sw_rescue = 0
    for result_row, status in qual_results:
        if status == 'ok':
            classC_rows.append(result_row)
            n_qualify += 1
            if result_row.get('_sw_rescue'):
                n_sw_rescue += 1
            if result_row.get('_c1'):
                n_c1_ok += 1
            else:
                n_c2_ok += 1
        elif status == 'no_pae':
            n_no_pae += 1
        elif status == 'no_domain':
            n_no_domain += 1
        elif status == 'domain_too_small':
            n_too_small += 1
        elif status == 'domain_low_plddt':
            n_low_plddt += 1
        elif status == 'no_pdb':
            n_no_pdb += 1

    print(f'  Qualifying:         {n_qualify}  (C1={n_c1_ok}, C2={n_c2_ok}, SW rescues={n_sw_rescue})')
    print(f'  No PAE file:        {n_no_pae}')
    print(f'  No domain:          {n_no_domain}')
    if n_too_small:
        print(f'  Domain too small:   {n_too_small}  (<{args.min_domain_size} residues → Class D)')
    if n_low_plddt:
        print(f'  Domain low pLDDT:   {n_low_plddt}  (<{args.min_domain_plddt:.0f} mean → Class D)')
    if n_no_pdb:
        print(f'  No PDB (ESM):       {n_no_pdb}')

    # Write candidate alignment table
    if args.write_candidates:
        all_rows_for_status = list(merged.iter_rows(named=True))
        _status_map = {}
        for result_row, status in qual_results:
            if result_row:
                _status_map[result_row['qseqid']] = status
        cand_pq = os.path.join(indir, 'classC_candidates.pq')
        cand_df = merged.with_columns(
            pl.col('qseqid').replace(_status_map, default='skip').alias('classC_status')
        )
        cand_df.write_parquet(cand_pq)
        print(f'  Candidates written: {cand_pq}')

    classC_ids = {r['qseqid'] for r in classC_rows}

    # ── [C-4] Build Class C structures ────────────────────────────────────────
    _fill_backend = _resolve_backend(args.backend)
    backend_label = 'MLX MiniFold' if _fill_backend else 'PyTorch MiniFold'
    mode_label = f'complete ({backend_label})' if args.fill_missing else 'domain'
    _write_status(f'Building {len(classC_rows)} Class C structures...')
    print(f'\n[C-4/4] Building {len(classC_rows)} Class C {mode_label} structures...')

    # Set up fragment kmer search globals (used by _frag_template_search inside build_complete_structure)
    global _SEARCH_BIN, _AFDB_SIDX_PATH, _AFDB_PQ_PATH, _ESM_SIDX_PATH, _ESM_PQ_PATH, _FRAG_PDB_DIR
    _FRAG_PDB_DIR = pdb_dir
    _afdb_dir = os.environ.get('AT_AFDB_DIR', '')
    if _afdb_dir:
        _sidx = os.path.join(_afdb_dir, 'afdb_v6_reps.sidx')
        _pq   = os.path.join(_afdb_dir, 'afdb_v6_reps.pq')
        if os.path.exists(_sidx) and os.path.exists(_pq):
            _sketch_cache = os.path.join(
                os.path.expanduser('~'), '.cache', 'alphatracer', 'dayhoff_sketch', 'release')
            _sb = os.path.join(_sketch_cache, 'search')
            if os.path.exists(_sb):
                _SEARCH_BIN     = _sb
                _AFDB_SIDX_PATH = _sidx
                _AFDB_PQ_PATH   = _pq
    _esm_dir = os.environ.get('AT_ESM_DIR',
                               str(os.path.join(os.path.expanduser('~'), 'Science', 'Data', 'ESMAtlas')))
    _esm_sidx = os.path.join(_esm_dir, 'esm_plddt70_non-afdb_reps.sidx')
    _esm_pq   = os.path.join(_esm_dir, 'esm_plddt70_non-afdb_reps.pq')
    if _SEARCH_BIN and os.path.exists(_esm_sidx) and os.path.exists(_esm_pq):
        _ESM_SIDX_PATH = _esm_sidx
        _ESM_PQ_PATH   = _esm_pq

    if args.fill_missing:
        _write_status('Loading ESM2-3B + MiniFold model (Class C fill-missing)...')
        _load_fold_models(model_size=args.mini_model_size,
                          compile_miniformer=not args.no_compile,
                          backend=args.backend)
        _write_status('Building Class C structures (fill-missing)...')

    n_ok_C = n_fail_C = 0
    timings_C = []
    failure_reasons_C = {}

    # Reuse the ref_poly cache already built in C-3 (no redundant gemmi reads)

    def _build_one_C(row):
        qseqid      = row['qseqid']
        best_domain = row['_best_domain']
        out_pdb     = os.path.join(outdir_C, f'classC:{qseqid}.pdb')

        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            return qseqid, True, None, 0.0, len(best_domain), True  # cached

        afdb_id = _A.get_afdb_id(row['sseqid'])
        if afdb_id:
            ref_poly = _ref_poly_cache.get(afdb_id)
        else:
            ph = row.get('protein_hash') or row['sseqid'].split('|')[0]
            ref_poly = _ref_poly_cache.get(ph)
        if ref_poly is None:
            return qseqid, False, 'ref PDB not found', 0.0, 0, False
        _, ops = _B.parse_alignment_ops(row['qseq_alg'], row['sseq_alg'])

        if args.fill_missing:
            safe_id  = re.sub(r'[^\w.-]', '_', qseqid)
            work_dir = os.path.join(outdir_C, f'_tmp_{safe_id}')
            ok, err, elapsed = build_complete_structure(
                best_domain, ops, ref_poly, row['full_qseq'], out_pdb, work_dir,
                mm_iters=args.mm_iters, ccd_iters=args.ccd_iters,
                ccd_tol=args.ccd_tol, n_flank=args.flank,
                min_frag_len=args.min_frag_len, lbfgsb_iters=args.lbfgsb_iters,
                model_size=args.mini_model_size, anchor_k=args.anchor_k,
                plddt_threshold=args.plddt_threshold,
                max_recyclings=args.max_recyclings,
                min_recycle_plddt=args.min_recycle_plddt,
                batch_tokens=args.batch_tokens,
                max_seq_len=args.max_seq_len,
                loop_closer=args.loop_closer,
            )
        else:
            ok, err, elapsed = build_domain_pdb(
                best_domain, ops, ref_poly, out_pdb,
                args.mm_iters, args.ccd_iters, args.ccd_tol, args.flank,
                anchor_k=args.anchor_k, loop_closer=args.loop_closer,
            )
            if ok:
                _split_broken_chains(out_pdb)
        return qseqid, ok, err, elapsed, len(best_domain), False

    # fill_missing uses MLX (not thread-safe) → must run in main thread (plain loop);
    # without fill_missing, use ThreadPoolExecutor for parallel domain building.
    from concurrent.futures import as_completed as _as_completed

    def _handle_C_result(qseqid, ok, err, elapsed, dom_len, cached):
        nonlocal n_ok_C, n_fail_C
        if cached:
            n_ok_C += 1
        elif ok:
            print(f'  {qseqid}: OK  {elapsed:.2f}s  (domain {dom_len} residues)')
            n_ok_C += 1
            timings_C.append(elapsed)
        else:
            print(f'  {qseqid}: FAILED — {err}')
            failure_reasons_C[qseqid] = err or ''
            n_fail_C += 1

    if args.fill_missing:
        # Must stay on main thread so MLX GPU stream is valid
        for row in classC_rows:
            _handle_C_result(*_build_one_C(row))
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = {ex.submit(_build_one_C, row): row['qseqid'] for row in classC_rows}
            for fut in _as_completed(futs):
                _handle_C_result(*fut.result())

    # Write classC outputs
    table_rows = []
    for row in classC_rows:
        qseqid      = row['qseqid']
        out_pdb     = os.path.join(outdir_C, f'classC:{qseqid}.pdb')
        afdb_id     = _A.get_afdb_id(row['sseqid'])
        ref_pdb     = _A.afdb_local_pdb(afdb_id, pdb_dir) if afdb_id else ''
        ok          = os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0
        best_domain = row['_best_domain']
        table_rows.append({
            'query_id':     qseqid,
            'built_pdb':    out_pdb,
            'ref_pdb':      ref_pdb,
            'domain_size':  str(len(best_domain)),
            'domain_start': str(min(best_domain)),
            'domain_end':   str(max(best_domain)),
            'status':       'ok' if ok else 'failed',
            'reason':       '' if ok else failure_reasons_C.get(qseqid, 'skipped'),
        })
    table_C_path = os.path.join(indir, 'classC_pdb_table.tsv')
    pl.DataFrame(table_rows).write_csv(table_C_path, separator='\t')
    pl.DataFrame([{'qseqid': r['qseqid']} for r in classC_rows]).write_parquet(
        os.path.join(indir, 'classC.pq'))

    print()
    print('=' * 60)
    print('Class C complete.')
    print(f'  Qualifying:     {n_qualify}')
    print(f'  PDBs written:   {n_ok_C}')
    print(f'  PDBs failed:    {n_fail_C}')
    if timings_C:
        print(f'  Wall time/protein: mean={sum(timings_C)/len(timings_C):.2f}s  '
              f'min={min(timings_C):.2f}s  max={max(timings_C):.2f}s')
    print('=' * 60)

    # ── CLASS D ───────────────────────────────────────────────────────────────
    if args.no_classD:
        print('\nClass D skipped (--no-classD).')
        return

    # Only treat a sequence as covered if it has a successfully written output PDB.
    # Sequences that qualified for A/B/C but whose structure build failed are
    # forwarded to Class D for de-novo prediction.
    outdir_A = os.path.join(indir, 'output_pdbs_classA')
    outdir_B = os.path.join(indir, 'output_pdbs_classB')

    def _has_pdb(outdir, prefix, qseqid):
        p = os.path.join(outdir, f'{prefix}:{qseqid}.pdb')
        return os.path.exists(p) and os.path.getsize(p) > 0

    covered_A = {sid for sid in classA_ids if _has_pdb(outdir_A, 'classA', sid)}
    covered_B = {sid for sid in classB_ids if _has_pdb(outdir_B, 'classB', sid)}
    covered_C = {sid for sid in classC_ids
                 if _has_pdb(outdir_C, 'classC', sid)}
    covered_ids = covered_A | covered_B | covered_C

    failed_A = classA_ids - covered_A
    failed_B = classB_ids - covered_B
    failed_C = classC_ids - covered_C
    n_forwarded = len(failed_A) + len(failed_B) + len(failed_C)

    filtered_fasta = os.path.join(indir, 'input_seqs_filtered.fa')
    if not os.path.exists(filtered_fasta):
        print(f'\nClass D: cannot find {filtered_fasta} — skipping.')
        return

    from alphatracer.utils.afdb_fetch import parse_fasta as _parse_fasta
    all_seqs   = {r.id: r.seq for r in _parse_fasta(filtered_fasta)}
    classD_seqs = {sid: seq for sid, seq in all_seqs.items()
                   if sid not in covered_ids}

    print(f'\n{"=" * 60}')
    print('AlphaTracer 1.0  —  Class D  (pLDDT-gated MLX MiniFold prediction)')
    print(f'{"=" * 60}')
    print(f'  Total filtered:   {len(all_seqs)}')
    print(f'  Covered A/B/C:    {len(covered_ids)}  '
          f'(A={len(covered_A)}, B={len(covered_B)}, C={len(covered_C)})')
    if n_forwarded:
        print(f'  Forwarded to D:   {n_forwarded}  '
              f'(A failures={len(failed_A)}, B={len(failed_B)}, C={len(failed_C)})')
    print(f'  Class D:          {len(classD_seqs)}')
    with open(os.path.join(indir, '.classD_total'), 'w') as _f:
        _f.write(str(len(classD_seqs)))

    if not classD_seqs:
        print('  No Class D sequences.')
        return

    if args.classD_limit > 0:
        classD_seqs = dict(list(classD_seqs.items())[:args.classD_limit])
        print(f'  (limited to first {args.classD_limit})')
    print()

    os.makedirs(outdir_D, exist_ok=True)
    _write_status('Loading ESM2-3B + MiniFold model (Class D)...')
    _load_fold_models(model_size=args.mini_model_size, backend=args.backend)
    _write_status('Predicting Class D structures...')

    n_ok_D = n_fail_D = 0
    timings_D = []
    D_rows    = []
    total_D   = len(classD_seqs)

    # ── Separate already-done and to-predict ──────────────────────────────────
    pending    = []   # (seq_id, masked_seq, max_r_for_seq)
    sec_pos_map = {}  # seq_id → list of 0-based SEC positions (for PDB restoration)
    for seq_id, seq in classD_seqs.items():
        out_pdb = os.path.join(outdir_D, f'classD:{seq_id}.pdb')
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            n_ok_D += 1
            D_rows.append({'query_id': seq_id, 'len': len(seq), 'status': 'ok',
                           'pdb': out_pdb, 'recyclings': '', 'plddt': '',
                           'elapsed_s': ''})
            continue
        if args.max_seq_len > 0 and len(seq) > args.max_seq_len:
            print(f'  [{seq_id}] SKIP: len={len(seq)} > --max-seq-len={args.max_seq_len}',
                  flush=True)
            n_fail_D += 1
            D_rows.append({'query_id': seq_id, 'len': len(seq), 'status': 'too_long',
                           'pdb': '', 'recyclings': '', 'plddt': '', 'elapsed_s': ''})
            continue
        masked_seq, sec_pos = _mask_selenocysteine(seq)
        if sec_pos:
            sec_pos_map[seq_id] = sec_pos
        pending.append((seq_id, masked_seq, args.max_recyclings))

    pending.sort(key=lambda x: len(x[1]))   # shorter first → tighter padding

    # ── Adaptive multi-round batched prediction ───────────────────────────────
    # Round r=0: predict all with num_recycling=0.
    # Sequences with mean_plddt < plddt_threshold proceed to round r=1, etc.
    batch_tokens  = args.batch_tokens
    plddt_thresh  = args.plddt_threshold
    max_global_r  = args.max_recyclings

    still_pending = pending   # list of (seq_id, seq, max_r)
    done_count    = 0

    print(f'  [D] Adaptive recycling: {len(pending)} seqs, '
          f'plddt_threshold={plddt_thresh}, max_recyclings={max_global_r}',
          flush=True)

    import traceback as _tb

    for r in range(max_global_r + 1):
        if not still_pending:
            break

        recycling_label = 'no recycling' if r == 0 else f'{r} recycling{"s" if r > 1 else ""}'
        _write_status(
            f'Predicting Class D structures — round {r}/{max_global_r} '
            f'({len(still_pending)} seqs, {recycling_label})...'
        )

        # Pack into batches by B × max_len² token budget (memory scales with both)
        batches, cur, cur_max = [], [], 0
        for item in still_pending:
            seq_id, seq, _ = item
            L = len(seq)
            new_max = max(cur_max, L)
            if cur and (len(cur) + 1) * new_max * new_max > batch_tokens:
                batches.append(cur); cur, cur_max = [], 0; new_max = L
            cur.append(item); cur_max = new_max
        if cur:
            batches.append(cur)

        print(f'  [D round {r}/{max_global_r}] {len(still_pending)} seqs → '
              f'{len(batches)} batches (recyclings={r})', flush=True)

        next_pending = []
        for b_idx, batch in enumerate(batches, 1):
            lens_in_batch = [len(seq) for _, seq, _ in batch]
            print(f'  Batch {b_idx}/{len(batches)}  B={len(batch)}  '
                  f'lens={lens_in_batch}  recyclings={r}', flush=True)
            t_batch = time.perf_counter()
            elapsed_per = 0.0
            try:
                batch_results = _fold_predict_batch(
                    [(sid, seq) for sid, seq, _ in batch], r)
                elapsed_per = (time.perf_counter() - t_batch) / len(batch)
                for (seq_id, pdb_str, mean_plddt), (_, seq, max_r) in zip(batch_results, batch):
                    done_count += 1
                    _prog = (f'  Class D: {done_count}/{len(pending)}  {seq_id}  '
                             f'len={len(seq)}  plddt={mean_plddt:.2f}  r={r}')
                    print(f'{_prog:<80}', end='\r', flush=True)
                    floor_hit = (r == 0 and mean_plddt < args.min_recycle_plddt)
                    if mean_plddt >= plddt_thresh or r >= max_r or floor_hit:
                        out_pdb = os.path.join(outdir_D, f'classD:{seq_id}.pdb')
                        pdb_str = _restore_selenocysteine(pdb_str, sec_pos_map.get(seq_id, []))
                        with open(out_pdb, 'w') as fh:
                            fh.write(pdb_str)
                        n_ok_D += 1
                        timings_D.append(elapsed_per)
                        if floor_hit:
                            print()  # end the \r line before warning
                            print(f'      → pLDDT {mean_plddt:.1f} < floor '
                                  f'{args.min_recycle_plddt:.0f}, skipping recycling',
                                  flush=True)
                        D_rows.append({
                            'query_id':   seq_id,
                            'len':        len(seq),
                            'status':     'ok',
                            'pdb':        out_pdb,
                            'recyclings': str(r),
                            'plddt':      f'{mean_plddt:.2f}',
                            'elapsed_s':  f'{elapsed_per:.1f}',
                        })
                    else:
                        next_pending.append((seq_id, seq, max_r))
                        done_count -= 1   # will be counted again next round
                print()  # finalize the \r progress line after each batch
            except Exception as exc:
                _tb.print_exc()
                elapsed_per = (time.perf_counter() - t_batch) / max(len(batch), 1)
                print()  # end any partial \r line before error output
                print(f'  Batch FAILED: {exc}', flush=True)
                for seq_id, seq, _ in batch:
                    n_fail_D += 1
                    D_rows.append({
                        'query_id': seq_id, 'len': len(seq), 'status': 'failed',
                        'pdb': '', 'recyclings': str(r),
                        'plddt': '', 'elapsed_s': f'{elapsed_per:.1f}',
                    })

        still_pending = next_pending

    table_D_path = os.path.join(indir, 'classD_pdb_table.tsv')
    pl.DataFrame(D_rows).write_csv(table_D_path, separator='\t')

    print()
    print('=' * 60)
    print('Class D complete.')
    print(f'  Sequences:      {total_D}')
    print(f'  PDBs written:   {n_ok_D}')
    print(f'  PDBs failed:    {n_fail_D}')
    if timings_D:
        print(f'  Wall time/protein: mean={sum(timings_D)/len(timings_D):.2f}s  '
              f'min={min(timings_D):.2f}s  max={max(timings_D):.2f}s')
    print(f'  Table:  {table_D_path}')
    print(f'  Output: {outdir_D}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
