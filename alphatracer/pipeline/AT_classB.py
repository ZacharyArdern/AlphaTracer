#!/usr/bin/env python3
"""
AT_classB.py  —  AlphaTracer 1.0, Class B pipeline

No PDBFixer, no hydrogen atoms, no side chains.

Structure building
------------------
  1. Backbone atoms (N, CA, C, O, CB) grafted from AlphaFold reference for
     matched/substituted positions; deleted residues simply omitted.
  2. Inserted residues placed in extended conformation (NeRF).
  3. Cyclic Coordinate Descent (CCD) closes backbone gaps at every indel site.
  4. OpenMM minimises with a backbone-only harmonic force field (bonds + angles
     + omega torsions).  No H atoms, no electrostatics, no VdW.

Usage
-----
  python AT_classB.py -i AT_processing_<name>/ [-t 4]
                      [--max-indels 3] [--max-indel-len 5] [--mm-iters 300]
"""

import os
import re
import sys
import time
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl
import parasail
import gemmi

from alphatracer.utils.afdb_fetch import (
    AFDB_VERSION, get_afdb_id, afdb_local_pdb, is_valid_pdb, fetch_afdb_pdbs,
)
from alphatracer.utils.esm_atlas_fetch import esm_local_pdb, _ESM_DIR, fetch_esm_structures
from openmm import (System, HarmonicBondForce, HarmonicAngleForce,
                    PeriodicTorsionForce, LangevinMiddleIntegrator, Platform)
from openmm.app import Topology, Simulation, PDBFile, Element
import openmm.unit as unit


# ── Constants ─────────────────────────────────────────────────────────────────

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}

# AFDB_VERSION imported from alphatracer.utils.afdb_fetch

# Ideal backbone geometry (Å and radians)
_BL = {'C-N': 1.335, 'N-CA': 1.460, 'CA-C': 1.522, 'C-O': 1.229, 'CA-CB': 1.526}
_BA = {
    'CA-C-N':  np.deg2rad(116.6),
    'C-N-CA':  np.deg2rad(121.9),
    'N-CA-C':  np.deg2rad(111.2),
    'CA-C-O':  np.deg2rad(120.8),
    'N-CA-CB': np.deg2rad(110.1),
    'C-CA-CB': np.deg2rad(110.1),
    'O-C-N':   np.deg2rad(122.6),
}
_PHI_EXT   = np.deg2rad(-120.0)
_PSI_EXT   = np.deg2rad(135.0)
_OMEGA     = np.deg2rad(180.0)
_PHI_HELIX = np.deg2rad(-57.0)   # alpha-helix canonical phi
_PSI_HELIX = np.deg2rad(-47.0)   # alpha-helix canonical psi

# Backbone-only OpenMM force field parameters
# Bonds: (r0_nm, k_kJ_mol_nm2)
_BOND_FF = {
    ('N',  'CA'): (0.1460, 282550.0),
    ('CA', 'C'):  (0.1522, 305450.0),
    ('C',  'N'):  (0.1335, 398410.0),   # inter-residue peptide bond
    ('C',  'O'):  (0.1229, 548780.0),
    ('CA', 'CB'): (0.1526, 305450.0),
}
# Angles: (theta0_rad, k_kJ_mol_rad2)
_ANGLE_FF = {
    ('N',  'CA', 'C'):  (_BA['N-CA-C'],  418.4),
    ('CA', 'C',  'N'):  (_BA['CA-C-N'],  585.8),
    ('C',  'N',  'CA'): (_BA['C-N-CA'],  418.4),
    ('CA', 'C',  'O'):  (_BA['CA-C-O'],  669.4),
    ('N',  'CA', 'CB'): (_BA['N-CA-CB'], 418.4),
    ('CB', 'CA', 'C'):  (_BA['C-CA-CB'], 418.4),
    ('O',  'C',  'N'):  (_BA['O-C-N'],   669.4),
}
_OMEGA_K = 9.6    # kJ/mol, omega (peptide plane) torsion force constant
_ATOM_MASS = {'N': 14.007, 'CA': 12.011, 'C': 12.011, 'O': 15.999, 'CB': 12.011}
_ATOM_ELEM = {'N': 'N',    'CA': 'C',    'C': 'C',    'O': 'O',    'CB': 'C'}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='AlphaTracer 1.0 — Class B pipeline (CCD + backbone OpenMM)'
    )
    p.add_argument('-i', '--input-dir', required=True,
                   help='Processing directory produced by AT_classA.py')
    p.add_argument('-t', '--threads', type=int, default=4)
    p.add_argument('--max-indels',          type=int, default=3)
    p.add_argument('--max-indel-len',       type=int, default=5)
    p.add_argument('--max-loop-indels',     type=int, default=8,
                   help='Max indels allowed when all fall in loop/coil context (default: 8)')
    p.add_argument('--max-loop-indel-len',  type=int, default=20,
                   help='Max length per indel in loop/coil context (default: 20)')
    p.add_argument('--min-pctsim',     type=float, default=40.0)
    p.add_argument('--mm-iters',      type=int,   default=300,
                   help='OpenMM minimisation iterations (default: 300)')
    p.add_argument('--ccd-iters',     type=int,   default=200,
                   help='Max CCD iterations per gap (default: 200)')
    p.add_argument('--ccd-tol',       type=float, default=0.15,
                   help='CCD convergence tolerance in Å (default: 0.15)')
    p.add_argument('--flank',         type=int,   default=3,
                   help='Mobile flanking residues for CCD (default: 3)')
    p.add_argument('--loop-closer',   default='ccd', choices=['ccd', 'promod3'],
                   help='Loop closing backend: ccd (default) or promod3')
    p.add_argument('--promod3-data-dir', default=None,
                   help='ProMod3 database directory (overrides PROMOD3_SHARED_DATA_PATH)')
    p.add_argument('--limit',         type=int,   default=0,
                   help='Process only first N Class B structures (0 = all)')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--c-alpha', action='store_true', default=False,
                      help='Output CA-only trace (no CCD, no OpenMM). Default mode.')
    mode.add_argument('--full-pdbs', action='store_true', default=False,
                      help='Output full backbone (N/CA/C/O/CB) with CCD and OpenMM minimisation.')
    return p.parse_args()


# get_afdb_id, afdb_local_pdb, esm_local_pdb (_esm_local_pdb alias below),
# _ESM_DIR, is_valid_pdb (_is_valid_pdb alias below), fetch_afdb_pdbs,
# fetch_esm_structures — all imported from alphatracer.utils.afdb_fetch / esm_atlas_fetch

_HAS_DB_TYPE = False  # overridden at runtime by auto-detection from allhits.pq

# Aliases for backward-compat with internal calls in this module
_esm_local_pdb = esm_local_pdb
_is_valid_pdb  = is_valid_pdb


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_nw(qseqid, sseqid, qseq, sseq):
    try:
        r = parasail.nw_trace_striped_16(qseq, sseq, 10, 1, parasail.blosum45)
        return qseqid, sseqid, r.traceback.query, r.traceback.ref, \
               r.traceback.comp.replace(' ', '-')
    except Exception as e:
        print(f'  Alignment failed for {qseqid}: {e}')
        return None


def classify_classB(qseq_alg, sseq_alg, max_indels, max_indel_len,
                    max_loop_indels=8, max_loop_indel_len=20):
    """Return True if this alignment qualifies for Class B.

    Strict path  : ≤max_indels indels each ≤max_indel_len aa.
    Relaxed path : ≤max_loop_indels indels each ≤max_loop_indel_len aa.
                   These must additionally pass an all-loop SS check at build
                   time (_indel_ss_context returns non-E and non-H for every
                   indel); beta-strand and helix-flanked indels are rejected
                   then.  Terminal query gaps are excluded from both counts.
    """
    ref_gaps = [len(m.group()) for m in re.finditer(r'-+', sseq_alg)]
    qry_gaps = [len(m.group()) for m in re.finditer(r'-+', qseq_alg.strip('-'))]
    all_gaps = ref_gaps + qry_gaps
    if not all_gaps:
        return False
    n, mx = len(all_gaps), max(all_gaps)
    if n <= max_indels and mx <= max_indel_len:
        return True
    if n <= max_loop_indels and mx <= max_loop_indel_len:
        return True
    return False


def parse_alignment_ops(qseq_alg, sseq_alg):
    """Return (ref_offset, ops). ops entries:
      ('match',     ref_pos, query_aa)
      ('deletion',  ref_pos, count)
      ('insertion', query_aas_str)
    """
    q_start = len(qseq_alg) - len(qseq_alg.lstrip('-'))
    q_end   = len(qseq_alg.rstrip('-')) or len(qseq_alg)
    ref_offset = sum(1 for c in sseq_alg[:q_start] if c != '-')
    ops = []
    rp = ref_offset
    i  = q_start
    while i < q_end:
        q, s = qseq_alg[i], sseq_alg[i]
        if q != '-' and s != '-':
            ops.append(('match', rp, q)); rp += 1; i += 1
        elif q == '-':
            ds = rp
            while i < q_end and qseq_alg[i] == '-': rp += 1; i += 1
            ops.append(('deletion', ds, rp - ds))
        else:
            # left_rpos: last consumed ref position; right_rpos: next to be consumed
            left_rpos = rp - 1
            ins = []
            while i < q_end and sseq_alg[i] == '-': ins.append(qseq_alg[i]); i += 1
            ops.append(('insertion', ''.join(ins), left_rpos, rp))
    return ref_offset, ops


# ── Secondary structure from backbone dihedrals ───────────────────────────────

def _dihedral(p0, p1, p2, p3):
    """Dihedral angle in degrees for atoms p0–p1–p2–p3."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def compute_ref_ss(ref_poly):
    """Compute 3-state secondary structure (H/E/C) for each residue in a gemmi
    polymer residue list using backbone phi/psi Ramachandran regions.

    Assignments:
      H  alpha-helix  : -100 <= phi <= -30  AND  -80 <= psi <= 0
      E  beta-strand  : phi <= -40          AND  psi >= 90
      C  coil / loop  : everything else

    Proline, and any residue missing phi or psi, is always assigned C.
    Returns a list of length len(ref_poly).
    """
    def _xyz(res, aname):
        try:
            a = res.find_atom(aname, '\0')
            if a:
                return np.array([a.pos.x, a.pos.y, a.pos.z])
        except Exception:
            pass
        return None

    n  = len(ref_poly)
    ss = ['C'] * n
    for i in range(n):
        if ref_poly[i].name == 'PRO':   # proline breaks helices
            continue

        phi = psi = None

        if i > 0:
            C_prev = _xyz(ref_poly[i - 1], 'C')
            N_i    = _xyz(ref_poly[i],     'N')
            CA_i   = _xyz(ref_poly[i],     'CA')
            C_i    = _xyz(ref_poly[i],     'C')
            if all(x is not None for x in (C_prev, N_i, CA_i, C_i)):
                phi = _dihedral(C_prev, N_i, CA_i, C_i)

        if i < n - 1:
            N_i    = _xyz(ref_poly[i],     'N')
            CA_i   = _xyz(ref_poly[i],     'CA')
            C_i    = _xyz(ref_poly[i],     'C')
            N_next = _xyz(ref_poly[i + 1], 'N')
            if all(x is not None for x in (N_i, CA_i, C_i, N_next)):
                psi = _dihedral(N_i, CA_i, C_i, N_next)

        if phi is None or psi is None:
            continue   # terminal residue — stays 'C'

        if -100 <= phi <= -30 and -80 <= psi <= 0:
            ss[i] = 'H'
        elif phi <= -40 and psi >= 90:
            ss[i] = 'E'

    return ss


def _indel_ss_context(op_idx, ops, ref_ss):
    """Return the secondary-structure context ('H', 'E', or 'C') for an indel op.

    Deletion  ('deletion',  ds, count)           : flanks are ref pos ds-1 and ds+count.
    Insertion ('insertion', aas, left_rp, right_rp): flanks stored in tuple.

    Returns 'H' if both flanks are helix, 'E' if either flank is beta-strand,
    'C' otherwise.
    """
    op = ops[op_idx]
    n  = len(ref_ss)

    def safe_ss(rpos):
        if rpos is None or rpos < 0 or rpos >= n:
            return 'C'
        return ref_ss[rpos]

    if op[0] == 'deletion':
        _, ds, count = op
        left_ss  = safe_ss(ds - 1)
        right_ss = safe_ss(ds + count)
    elif op[0] == 'insertion':
        # op = ('insertion', aas, left_rpos, right_rpos)
        left_ss  = safe_ss(op[2])
        right_ss = safe_ss(op[3])
    else:
        return 'C'

    if left_ss == 'H' and right_ss == 'H':
        return 'H'
    if left_ss == 'E' and right_ss == 'E':
        return 'E'
    return 'C'


# ── NeRF backbone geometry ────────────────────────────────────────────────────

def place_atom(a, b, c, bond_len, angle_rad, dihedral_rad):
    """Place D given A, B, C: |CD|=bond_len, angle B-C-D=angle_rad,
    dihedral A-B-C-D=dihedral_rad.  All positions in Å."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    bc = c - b;  bc /= np.linalg.norm(bc)
    n  = np.cross(b - a, bc)
    nl = np.linalg.norm(n)
    if nl > 1e-10:
        n /= nl
    else:
        perp = np.array([1., 0., 0.]) if abs(bc[0]) < 0.9 else np.array([0., 1., 0.])
        n = np.cross(perp, bc);  n /= np.linalg.norm(n)
    m = np.cross(bc, n)
    return c + bond_len * (
        -np.cos(angle_rad)  * bc
        + np.sin(angle_rad) * np.cos(dihedral_rad) * m
        + np.sin(angle_rad) * np.sin(dihedral_rad) * n
    )


def build_extended_residue(prev_N, prev_CA, prev_C, resname):
    """Return {atom_name: position} for one new residue in extended conformation."""
    N  = place_atom(prev_N,  prev_CA, prev_C, _BL['C-N'],  _BA['CA-C-N'], _PSI_EXT)
    CA = place_atom(prev_CA, prev_C,  N,      _BL['N-CA'], _BA['C-N-CA'], _OMEGA)
    C  = place_atom(prev_C,  N,       CA,     _BL['CA-C'], _BA['N-CA-C'], _PHI_EXT)
    O  = place_atom(N,       CA,      C,      _BL['C-O'],  _BA['CA-C-O'], np.pi)
    atoms = {'N': N, 'CA': CA, 'C': C, 'O': O}
    if resname != 'GLY':
        atoms['CB'] = place_atom(prev_C, N, CA, _BL['CA-CB'], _BA['N-CA-CB'],
                                 np.deg2rad(-121.5))
    return atoms


def build_helix_residue(prev_N, prev_CA, prev_C, resname):
    """Return {atom_name: position} for one new residue in alpha-helix conformation.

    Uses canonical alpha-helix phi=-57°, psi=-47° (NeRF argument order mirrors
    build_extended_residue: psi feeds the N placement, phi feeds C placement).
    """
    N  = place_atom(prev_N,  prev_CA, prev_C, _BL['C-N'],  _BA['CA-C-N'], _PSI_HELIX)
    CA = place_atom(prev_CA, prev_C,  N,      _BL['N-CA'], _BA['C-N-CA'], _OMEGA)
    C  = place_atom(prev_C,  N,       CA,     _BL['CA-C'], _BA['N-CA-C'], _PHI_HELIX)
    O  = place_atom(N,       CA,      C,      _BL['C-O'],  _BA['CA-C-O'], np.pi)
    atoms = {'N': N, 'CA': CA, 'C': C, 'O': O}
    if resname != 'GLY':
        atoms['CB'] = place_atom(prev_C, N, CA, _BL['CA-CB'], _BA['N-CA-CB'],
                                 np.deg2rad(-121.5))
    return atoms


def build_residue_from_angles(prev_N, prev_CA, prev_C, resname, phi, psi):
    """Build one residue using explicit phi/psi angles (radians)."""
    N  = place_atom(prev_N,  prev_CA, prev_C, _BL['C-N'],  _BA['CA-C-N'], psi)
    CA = place_atom(prev_CA, prev_C,  N,      _BL['N-CA'], _BA['C-N-CA'], _OMEGA)
    C  = place_atom(prev_C,  N,       CA,     _BL['CA-C'], _BA['N-CA-C'], phi)
    O  = place_atom(N,       CA,      C,      _BL['C-O'],  _BA['CA-C-O'], np.pi)
    atoms = {'N': N, 'CA': CA, 'C': C, 'O': O}
    if resname != 'GLY':
        atoms['CB'] = place_atom(prev_C, N, CA, _BL['CA-CB'], _BA['N-CA-CB'],
                                 np.deg2rad(-121.5))
    return atoms


# beta-bulge phi/psi for 1- and 2-residue insertions in strand context
# Classic bulge geometry from Richardson 1981 / Wouters & Curmi 1995
_BULGE_1 = [(-100.0,  30.0)]                          # Δ=1: one bulge residue
_BULGE_2 = [(-100.0, 120.0), (-130.0, 130.0)]         # Δ=2: classic bulge pair


def build_directed_residues(prev_N, prev_CA, prev_C, aas, target_N, ss_ctx='C'):
    """Build insertion residues aiming each CA along the straight-line path toward
    target_N.  Uses a 1-D scan over psi (fine grid) to choose N placement, which
    exactly determines CA given omega=180°.  phi is then chosen to minimise the
    C-to-target distance for the last residue, or set to strand/helix default.

    Returns list of (resname, atoms) tuples.
    """
    pN, pCA, pC = np.asarray(prev_N), np.asarray(prev_CA), np.asarray(prev_C)
    n = len(aas)
    target = np.asarray(target_N)
    result = []

    psi_grid = np.deg2rad(np.arange(-180, 180, 2))   # 180 points, fast
    phi_grid = np.deg2rad(np.arange(-180, 180, 5))   # used only for last residue

    for i, aa in enumerate(aas):
        rn = ONE_TO_THREE.get(aa, 'ALA')
        remaining = n - i

        # desired CA: interpolate toward target
        frac = 1.0 / remaining
        desired_CA = pCA + frac * (target - pCA)

        # scan psi to place N (and thus CA) closest to desired_CA
        # CA position depends only on psi (N position) since omega is fixed
        best_psi, best_d_ca = _PSI_EXT, np.inf
        for psi in psi_grid:
            N_try = place_atom(pN, pCA, pC, _BL['C-N'], _BA['CA-C-N'], psi)
            CA_try = place_atom(pCA, pC, N_try, _BL['N-CA'], _BA['C-N-CA'], _OMEGA)
            d = np.linalg.norm(CA_try - desired_CA)
            if d < best_d_ca:
                best_d_ca, best_psi = d, psi

        # for phi: on the last residue scan to minimise C-to-target; elsewhere use default
        if remaining == 1:
            best_phi, best_d_c = _PHI_EXT, np.inf
            N_best = place_atom(pN, pCA, pC, _BL['C-N'], _BA['CA-C-N'], best_psi)
            CA_best = place_atom(pCA, pC, N_best, _BL['N-CA'], _BA['C-N-CA'], _OMEGA)
            for phi in phi_grid:
                C_try = place_atom(pC, N_best, CA_best, _BL['CA-C'], _BA['N-CA-C'], phi)
                d = np.linalg.norm(C_try - target)
                if d < best_d_c:
                    best_d_c, best_phi = d, phi
        else:
            best_phi = _PHI_HELIX if ss_ctx == 'H' else _PHI_EXT

        atoms = build_residue_from_angles(pN, pCA, pC, rn, best_phi, best_psi)
        result.append((rn, atoms))
        pN, pCA, pC = atoms['N'], atoms['CA'], atoms['C']

    return result


# ── Numerical tripeptide closure ──────────────────────────────────────────────

def _tripeptide_close(prev_N, prev_CA, prev_C, aas, target_N, n_solutions=8):
    """Minimise gap between C of last residue and target_N over 6 torsion DOF.

    Uses multi-start L-BFGS-B; returns (list of (resname, atoms), gap_Å) or
    (None, gap_Å) if best gap > 2 Å.
    """
    from scipy.optimize import minimize

    assert len(aas) == 3
    resnames = [ONE_TO_THREE.get(aa, 'ALA') for aa in aas]
    target = np.asarray(target_N)

    pN0, pCA0, pC0 = np.asarray(prev_N), np.asarray(prev_CA), np.asarray(prev_C)

    def build_chain(x):
        phi1, psi1, phi2, psi2, phi3, psi3 = x
        a1 = build_residue_from_angles(pN0, pCA0, pC0, resnames[0], phi1, psi1)
        a2 = build_residue_from_angles(a1['N'], a1['CA'], a1['C'], resnames[1], phi2, psi2)
        a3 = build_residue_from_angles(a2['N'], a2['CA'], a2['C'], resnames[2], phi3, psi3)
        return a1, a2, a3

    def residual(x):
        _, _, a3 = build_chain(x)
        return float(np.sum((a3['C'] - target) ** 2))

    rng = np.random.default_rng(42)
    starts = [
        np.array([_PHI_EXT,   _PSI_EXT]   * 3),
        np.array([_PHI_HELIX, _PSI_HELIX] * 3),
        np.array([_PHI_EXT,   _PSI_HELIX, _PHI_EXT,   _PSI_EXT,   _PHI_EXT,   _PSI_EXT]),
        np.array([_PHI_HELIX, _PSI_EXT,   _PHI_HELIX, _PSI_HELIX, _PHI_EXT,   _PSI_EXT]),
    ]
    starts += [rng.uniform(-np.pi, np.pi, 6) for _ in range(n_solutions - 4)]

    best_res, best_x = np.inf, None
    for x0 in starts:
        r = minimize(residual, x0, method='L-BFGS-B',
                     options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-9})
        if r.fun < best_res:
            best_res, best_x = r.fun, r.x

    gap = np.sqrt(best_res)
    if best_x is None or gap > 2.0:
        return None, gap

    a1, a2, a3 = build_chain(best_x)
    return [(resnames[0], a1), (resnames[1], a2), (resnames[2], a3)], gap


def tripeptide_close_insertion(prev_N, prev_CA, prev_C, aas, target_N,
                               ss_ctx='C', n_solutions=8):
    """Build n≥3 insertion residues using directed geometry + tripeptide closure.

    For n==3: directly optimises all 6 torsions.
    For n>3: builds first n-3 residues with directed geometry, then tripeptide
             closure on the last 3.

    Returns (list of (resname, atoms), gap_Å), or (None, gap_Å) on failure.
    """
    n = len(aas)
    if n < 3:
        return None, 999.

    target = np.asarray(target_N)

    if n == 3:
        return _tripeptide_close(prev_N, prev_CA, prev_C, aas, target,
                                 n_solutions=n_solutions)

    pre_aas  = aas[:n - 3]
    trip_aas = aas[n - 3:]

    pre_residues = build_directed_residues(prev_N, prev_CA, prev_C, pre_aas,
                                           target, ss_ctx=ss_ctx)
    if not pre_residues:
        return None, 999.

    _, last_atoms = pre_residues[-1]
    pN2, pCA2, pC2 = last_atoms['N'], last_atoms['CA'], last_atoms['C']

    trip_result, gap = _tripeptide_close(pN2, pCA2, pC2, trip_aas, target,
                                         n_solutions=n_solutions)
    if trip_result is None:
        return None, gap

    return pre_residues + trip_result, gap


# ── CCD loop closure ──────────────────────────────────────────────────────────

def _rotation_matrix(axis, angle):
    """Rodrigues' rotation matrix around unit vector axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
    ])


def _ccd_step(positions, a_idx, b_idx, downstream, end_idx, target):
    """One CCD step: rotate downstream atoms around bond a→b to move
    positions[end_idx] toward target.  Modifies positions in-place."""
    a, b = positions[a_idx], positions[b_idx]
    u = b - a;  ul = np.linalg.norm(u)
    if ul < 1e-10: return
    u /= ul

    v = positions[end_idx] - a
    w = target - a
    v_p = v - np.dot(v, u) * u
    w_p = w - np.dot(w, u) * u
    A = np.dot(v_p, w_p)
    B = np.dot(np.cross(u, v_p), w_p)
    if abs(A) < 1e-10 and abs(B) < 1e-10: return

    theta = np.arctan2(B, A)
    R = _rotation_matrix(u, theta)
    for idx in downstream:
        d = positions[idx] - a
        positions[idx] = a + R @ d


def ccd_close(positions, pivot_bonds, end_idx, target,
              max_iter=200, tol=0.15):
    """CCD loop closure.

    positions   : flat list of np.array([x,y,z]) for all atoms
    pivot_bonds : list of (a_idx, b_idx, [downstream_indices])
    end_idx     : index of the atom that must reach `target`
    target      : np.array([x,y,z]) target position (N of anchor residue)

    Returns final distance to target (Å).
    """
    for _ in range(max_iter):
        dist = np.linalg.norm(positions[end_idx] - target)
        if dist <= tol:
            break
        for a, b, downstream in pivot_bonds:
            _ccd_step(positions, a, b, downstream, end_idx, target)
    return float(np.linalg.norm(positions[end_idx] - target))


# ── OpenMM backbone-only system ───────────────────────────────────────────────

def _atom_indices(residues):
    """Return flat list of (res_i, atom_name) pairs and index lookup dict."""
    flat = []
    idx  = {}
    for ri, res in enumerate(residues):
        for aname in ['N', 'CA', 'C', 'O', 'CB']:
            if aname in res['atoms']:
                idx[(ri, aname)] = len(flat)
                flat.append((ri, aname))
    return flat, idx


_VDW_R = {'N': 1.55, 'CA': 1.70, 'C': 1.70, 'O': 1.52, 'CB': 1.70}


def _has_backbone_clash(residues, threshold=0.72):
    """Return True if any non-bonded backbone atom pair is closer than threshold × (r1+r2).

    Skips adjacent atoms (bonded neighbours within the same or consecutive residues).
    Uses a numpy pairwise distance matrix for speed.
    """
    coords, radii = [], []
    for res in residues:
        for aname in ('N', 'CA', 'C', 'O', 'CB'):
            if aname in res['atoms']:
                coords.append(res['atoms'][aname])
                radii.append(_VDW_R.get(aname, 1.70))
    if len(coords) < 4:
        return False
    coords = np.array(coords)
    radii  = np.array(radii)
    diff   = coords[:, None, :] - coords[None, :, :]
    dist   = np.sqrt((diff * diff).sum(axis=-1))
    sum_r  = radii[:, None] + radii[None, :]
    # upper triangle only, skip i and i+1 (bonded neighbours)
    n    = len(coords)
    mask = np.triu(np.ones((n, n), dtype=bool), k=2)
    return bool(np.any(dist[mask] < threshold * sum_r[mask]))


def build_openmm_system(residues):
    """Build OpenMM System + Topology + positions for backbone-only minimisation.

    residues: list of {'resname': str, 'atoms': {atom_name: np.array Å}}
    Returns (system, topology, positions_nm).
    """
    flat, idx = _atom_indices(residues)
    n_atoms = len(flat)

    # ── System ────────────────────────────────────────────────────────────────
    system = System()
    for ri, aname in flat:
        system.addParticle(_ATOM_MASS[aname])

    # ── Bonds ─────────────────────────────────────────────────────────────────
    bf = HarmonicBondForce()
    bond_pairs = set()

    def add_bond(ri, a, rj, b):
        ia, ib = idx.get((ri, a)), idx.get((rj, b))
        if ia is None or ib is None: return
        key = (min(ia, ib), max(ia, ib))
        if key in bond_pairs: return
        bond_pairs.add(key)
        r0, k = _BOND_FF.get((a, b)) or _BOND_FF.get((b, a)) or (0.1500, 250000.0)
        bf.addBond(ia, ib, r0, k)

    for ri, res in enumerate(residues):
        add_bond(ri, 'N',  ri, 'CA')
        add_bond(ri, 'CA', ri, 'C')
        add_bond(ri, 'C',  ri, 'O')
        if 'CB' in res['atoms']:
            add_bond(ri, 'CA', ri, 'CB')
        if ri + 1 < len(residues):
            add_bond(ri, 'C', ri + 1, 'N')   # peptide bond
    system.addForce(bf)

    # ── Angles ────────────────────────────────────────────────────────────────
    af = HarmonicAngleForce()

    def add_angle(ri, a, rj, b, rk, c):
        ia, ib, ic = idx.get((ri,a)), idx.get((rj,b)), idx.get((rk,c))
        if ia is None or ib is None or ic is None: return
        theta0, k = _ANGLE_FF.get((a, b, c)) or _ANGLE_FF.get((c, b, a)) or \
                    (np.deg2rad(109.5), 418.4)
        af.addAngle(ia, ib, ic, theta0, k)

    for ri, res in enumerate(residues):
        add_angle(ri,'N',  ri,'CA', ri,'C')
        add_angle(ri,'CA', ri,'C',  ri,'O')
        if 'CB' in res['atoms']:
            add_angle(ri,'N',  ri,'CA', ri,'CB')
            add_angle(ri,'CB', ri,'CA', ri,'C')
        if ri + 1 < len(residues):
            add_angle(ri,'CA', ri,'C',  ri+1,'N')
            add_angle(ri,'C',  ri+1,'N', ri+1,'CA')
            add_angle(ri,'O',  ri,'C',  ri+1,'N')
    system.addForce(af)

    # ── Omega torsions (peptide planarity) ────────────────────────────────────
    tf = PeriodicTorsionForce()
    for ri in range(len(residues) - 1):
        ia = idx.get((ri,   'CA'))
        ib = idx.get((ri,   'C'))
        ic = idx.get((ri+1, 'N'))
        id_ = idx.get((ri+1, 'CA'))
        if None not in (ia, ib, ic, id_):
            tf.addTorsion(ia, ib, ic, id_, 2, np.pi, _OMEGA_K)
    system.addForce(tf)

    # ── Topology (for PDB output) ─────────────────────────────────────────────
    top   = Topology()
    chain = top.addChain('A')
    atoms_top = []
    prev_ri = -1
    cur_res  = None
    for ri, aname in flat:
        if ri != prev_ri:
            cur_res = top.addResidue(residues[ri]['resname'], chain)
            prev_ri = ri
        elem = Element.getBySymbol(_ATOM_ELEM[aname])
        atoms_top.append(top.addAtom(aname, elem, cur_res))
    for ia, ib in bond_pairs:
        top.addBond(atoms_top[ia], atoms_top[ib])

    # ── Positions (nm) ────────────────────────────────────────────────────────
    positions_nm = []
    for ri, aname in flat:
        pos_ang = residues[ri]['atoms'][aname]
        positions_nm.append((float(pos_ang[0])/10,
                              float(pos_ang[1])/10,
                              float(pos_ang[2])/10))

    return system, top, positions_nm


# ── Platform (initialised once) ───────────────────────────────────────────────

_PLATFORM = None

def _get_platform():
    global _PLATFORM
    if _PLATFORM is None:
        for name in ['HIP', 'CUDA', 'OpenCL', 'CPU']:
            try:
                _PLATFORM = Platform.getPlatformByName(name)
                print(f'  OpenMM platform: {_PLATFORM.getName()}')
                break
            except Exception:
                continue
        if _PLATFORM is None:
            _PLATFORM = Platform.getPlatformByName('Reference')
    return _PLATFORM


# ── CA-only build ────────────────────────────────────────────────────────────

def build_classB_ca_only(row, pdb_dir, out_pdb):
    """Build a CA-only trace for one Class B protein.

    Matched residues: CA copied directly from reference AFDB structure.
    Deleted residues: omitted.
    Inserted residues: CA linearly interpolated between flanking reference CAs.

    Returns (success, error_msg, elapsed_s).
    """
    t0 = time.perf_counter()
    try:
        qseqid   = row['qseqid']
        full_qseq = row['full_qseq']
        qseq_alg = row['qseq_alg']
        sseq_alg = row['sseq_alg']
        sseqid   = row['sseqid']

        db_type = row.get('db_type', 'afdb') or 'afdb'
        if db_type == 'esm_atlas':
            protein_hash = row.get('protein_hash') or sseqid.split('|')[0]
            src_pdb = _esm_local_pdb(protein_hash, pdb_dir)
            if not os.path.exists(src_pdb):
                return False, f'ESM Atlas PDB not cached ({protein_hash})', 0.
        else:
            afdb_id = get_afdb_id(sseqid)
            if afdb_id is None:
                return False, f'cannot parse AFDB id from "{sseqid}"', 0.
            src_pdb = afdb_local_pdb(afdb_id, pdb_dir)
            if not os.path.exists(src_pdb):
                return False, f'source PDB not found ({afdb_id})', 0.

        _, ops = parse_alignment_ops(qseq_alg, sseq_alg)

        st = gemmi.read_structure(src_pdb)
        if db_type == 'esm_atlas':
            st.setup_entities()
        ref_ch   = st[0]['A']
        ref_poly = [r for r in ref_ch if r.entity_type == gemmi.EntityType.Polymer]

        def _get_ca(res):
            try:
                a = res.find_atom('CA', '\0')
                if a:
                    return np.array([a.pos.x, a.pos.y, a.pos.z])
            except Exception:
                pass
            return None

        # First pass: collect entries as ('ref', resname, ca) or ('ins', [aas])
        entries = []
        for op in ops:
            if op[0] == 'match':
                _, rpos, qaa = op
                if rpos >= len(ref_poly):
                    return False, f'ref_pos {rpos} out of range', time.perf_counter() - t0
                ca = _get_ca(ref_poly[rpos])
                if ca is None:
                    return False, f'no CA at ref_pos {rpos}', time.perf_counter() - t0
                entries.append(('ref', ONE_TO_THREE.get(qaa, 'ALA'), ca))
            elif op[0] == 'deletion':
                pass
            elif op[0] == 'insertion':
                entries.append(('ins', op[1], None))

        # Second pass: interpolate inserted CA positions
        cas = []   # final list of (resname, ca_xyz)
        for i, entry in enumerate(entries):
            if entry[0] == 'ref':
                cas.append((entry[1], entry[2]))
            else:
                ins_aas = entry[1]
                n = len(ins_aas)
                prev_ca = next((entries[j][2] for j in range(i - 1, -1, -1)
                                if entries[j][0] == 'ref'), None)
                next_ca = next((entries[j][2] for j in range(i + 1, len(entries))
                                if entries[j][0] == 'ref'), None)
                for k, aa in enumerate(ins_aas):
                    frac = (k + 1) / (n + 1)
                    if prev_ca is not None and next_ca is not None:
                        ca = prev_ca + frac * (next_ca - prev_ca)
                    elif prev_ca is not None:
                        ca = prev_ca + (k + 1) * np.array([3.8, 0.0, 0.0])
                    elif next_ca is not None:
                        ca = next_ca - (n - k) * np.array([3.8, 0.0, 0.0])
                    else:
                        ca = np.array([0.0, 0.0, 0.0])
                    cas.append((ONE_TO_THREE.get(aa, 'ALA'), ca))

        if not cas:
            return False, 'no residues built', time.perf_counter() - t0

        with open(out_pdb, 'w') as f:
            for idx, (resname, ca) in enumerate(cas, 1):
                f.write(f'ATOM  {idx:5d}  CA  {resname} A{idx:4d}    '
                        f'{ca[0]:8.3f}{ca[1]:8.3f}{ca[2]:8.3f}  1.00  0.00           C\n')
            f.write('END\n')

        return True, None, time.perf_counter() - t0

    except Exception as e:
        return False, str(e), time.perf_counter() - t0


# ── Per-structure build ───────────────────────────────────────────────────────

def build_classB_structure(row, pdb_dir, out_pdb, mm_iters, ccd_iters,
                            ccd_tol, n_flank, loop_closer='ccd',
                            max_indels=3, max_indel_len=5):
    """Build one Class B backbone structure.

    Returns (success, error_msg, elapsed_s).
    loop_closer: 'ccd' (default) or 'promod3'
    """
    t0 = time.perf_counter()
    try:
        qseqid    = row['qseqid']
        full_qseq = row['full_qseq']
        qseq_alg  = row['qseq_alg']
        sseq_alg  = row['sseq_alg']
        sseqid    = row['sseqid']

        db_type = row.get('db_type', 'afdb') or 'afdb'
        if db_type == 'esm_atlas':
            protein_hash = row.get('protein_hash') or sseqid.split('|')[0]
            src_pdb = _esm_local_pdb(protein_hash, pdb_dir)
            if not os.path.exists(src_pdb):
                return False, f'ESM Atlas PDB not cached ({protein_hash})', 0.
        else:
            afdb_id = get_afdb_id(sseqid)
            if afdb_id is None:
                return False, f'cannot parse AFDB id from "{sseqid}"', 0.
            src_pdb = afdb_local_pdb(afdb_id, pdb_dir)
            if not os.path.exists(src_pdb):
                return False, f'source PDB not found ({afdb_id})', 0.

        # ── 1. Parse alignment ────────────────────────────────────────────────
        _, ops = parse_alignment_ops(qseq_alg, sseq_alg)

        # ── 2. Read reference backbone ────────────────────────────────────────
        st = gemmi.read_structure(src_pdb)
        if db_type == 'esm_atlas':
            st.setup_entities()
        ref_ch   = st[0]['A']
        ref_poly = [r for r in ref_ch if r.entity_type == gemmi.EntityType.Polymer]

        def ref_backbone(res):
            """Extract {atom_name: np.array} for backbone atoms."""
            atoms = {}
            for aname in ['N', 'CA', 'C', 'O', 'CB']:
                try:
                    a = res.find_atom(aname, '\0')
                    if a:
                        atoms[aname] = np.array([a.pos.x, a.pos.y, a.pos.z])
                except Exception:
                    pass
            return atoms

        # ── 2b. Indel secondary-structure context ─────────────────────────────
        ref_ss = compute_ref_ss(ref_poly)

        indel_contexts = {}
        for k, op in enumerate(ops):
            if op[0] in ('deletion', 'insertion'):
                indel_contexts[k] = _indel_ss_context(k, ops, ref_ss)

        # Reject any structure where an indel is flanked by beta-strand residues
        for k, op in enumerate(ops):
            if indel_contexts.get(k) == 'E':
                return False, (
                    f'{op[0]} flanked by beta-strand at ref context (E-rejection)'
                ), time.perf_counter() - t0

        # For relaxed-path candidates (exceed strict indel limits), also reject
        # if any indel is helix-flanked — only all-coil indels are safe to close.
        ref_gaps = [len(m.group()) for m in re.finditer(r'-+', sseq_alg)]
        qry_gaps = [len(m.group()) for m in re.finditer(r'-+', qseq_alg.strip('-'))]
        all_gaps = ref_gaps + qry_gaps
        n_ind, mx_ind = (len(all_gaps), max(all_gaps)) if all_gaps else (0, 0)
        is_relaxed = all_gaps and (n_ind > max_indels or mx_ind > max_indel_len)
        if is_relaxed:
            for k, op in enumerate(ops):
                if indel_contexts.get(k) == 'H':
                    return False, (
                        f'{op[0]} helix-flanked on relaxed path (H-rejection)'
                    ), time.perf_counter() - t0

        # ── 3. Build flat residue list ────────────────────────────────────────
        # Each entry: {'resname': str, 'atoms': dict, 'from_ref': bool}
        residues = []
        # Track gap sites: (before_res_idx, after_res_idx, gap_type)
        # gap_type: 'insertion' | 'deletion'
        gap_sites = []   # list of res_idx BEFORE the gap (in residues list)

        for op_idx, op in enumerate(ops):
            if op[0] == 'match':
                _, rpos, qaa = op
                if rpos >= len(ref_poly):
                    return False, f'ref_pos {rpos} out of range', 0.
                resname = ONE_TO_THREE.get(qaa, 'ALA')
                atoms   = ref_backbone(ref_poly[rpos])
                # Keep only backbone atoms valid for this residue
                keep = {'N', 'CA', 'C', 'O'} | ({'CB'} if qaa != 'G' else set())
                atoms = {k: v for k, v in atoms.items() if k in keep}
                if len({'N', 'CA', 'C', 'O'} & atoms.keys()) < 4:
                    return False, f'incomplete backbone at ref_pos {rpos}', 0.
                residues.append({'resname': resname, 'atoms': atoms, 'from_ref': True})

            elif op[0] == 'deletion':
                # Mark gap: the last residue so far needs to connect to the next
                if residues:
                    gap_sites.append((len(residues) - 1, 'deletion'))

            elif op[0] == 'insertion':
                ins_aas = op[1]   # op = ('insertion', aas, left_rpos, right_rpos)
                if not residues:
                    return False, 'insertion at chain start not supported', 0.

                # Mark that a gap exists before these inserted residues
                gap_sites.append((len(residues) - 1, 'insertion'))

                # Place inserted residues using helix angles if both flanking
                # reference residues are in a helix, otherwise extended conformation
                prev = residues[-1]['atoms']
                if not all(k in prev for k in ('N', 'CA', 'C')):
                    return False, 'cannot place insertion: prev residue missing backbone', 0.
                pN, pCA, pC = prev['N'], prev['CA'], prev['C']

                ss_ctx = indel_contexts.get(op_idx, 'C')
                res_builder = (build_helix_residue if ss_ctx == 'H'
                               else build_extended_residue)

                for aa in ins_aas:
                    resname = ONE_TO_THREE.get(aa, 'ALA')
                    new_atoms = res_builder(pN, pCA, pC, resname)
                    residues.append({'resname': resname, 'atoms': new_atoms,
                                     'from_ref': False})
                    pN, pCA, pC = new_atoms['N'], new_atoms['CA'], new_atoms['C']

        if not residues:
            return False, 'no residues built', 0.

        # Verify sequence
        built_seq = ''.join(
            next(k for k, v in ONE_TO_THREE.items() if v == r['resname'])
            for r in residues
        )
        if built_seq != full_qseq:
            return False, f'sequence mismatch after building', 0.

        # ── 4. CCD loop closure at each gap ───────────────────────────────────
        if gap_sites:
            # Build flat position array for CCD
            # Map (res_i, atom_name) → flat index
            flat_atoms  = []  # [(res_i, atom_name), ...]
            flat_pos    = []  # np.array positions
            atom_to_idx = {}

            for ri, res in enumerate(residues):
                for aname in ['N', 'CA', 'C', 'O', 'CB']:
                    if aname in res['atoms']:
                        atom_to_idx[(ri, aname)] = len(flat_atoms)
                        flat_atoms.append((ri, aname))
                        flat_pos.append(res['atoms'][aname].copy())

            for (gap_before_ri, gap_type) in gap_sites:
                # The residue AFTER the gap
                after_ri = gap_before_ri + 1
                if after_ri >= len(residues):
                    continue

                # Target: N of the first residue after the gap
                target_idx = atom_to_idx.get((after_ri, 'N'))
                if target_idx is None:
                    continue
                target = flat_pos[target_idx].copy()

                # Chain-end: C of the last residue before the gap
                # (which should be the last inserted residue for insertions,
                # or the residue just before the deletion)
                end_ri = gap_before_ri
                if gap_type == 'insertion':
                    # Find last inserted residue index
                    # gap_before_ri is the last reference residue; the inserted
                    # ones follow, but for the residues list after_ri is the
                    # first non-inserted → end_ri = after_ri - 1
                    end_ri = after_ri - 1

                end_idx = atom_to_idx.get((end_ri, 'C'))
                if end_idx is None:
                    continue

                # Mobile: inserted residues + up to n_flank residues on N side
                if gap_type == 'insertion':
                    n_inserted = after_ri - gap_before_ri - 1
                    mobile_start = max(0, gap_before_ri + 1 - n_flank)
                    mobile_end   = after_ri  # exclusive
                else:
                    # deletion: rotate n_flank residues before the gap
                    mobile_start = max(0, gap_before_ri + 1 - n_flank)
                    mobile_end   = gap_before_ri + 1

                # Build rotatable bonds: phi (N→CA) and psi (CA→C) for each mobile res
                pivot_bonds = []
                for ri in range(mobile_start, mobile_end):
                    for (a_name, b_name) in [('N', 'CA'), ('CA', 'C')]:
                        ia = atom_to_idx.get((ri, a_name))
                        ib = atom_to_idx.get((ri, b_name))
                        if ia is None or ib is None:
                            continue
                        # Downstream: all atoms at res >= ri that come after b in backbone order
                        downstream = [
                            atom_to_idx[(rj, an)]
                            for rj in range(ri, mobile_end)
                            for an in ['N', 'CA', 'C', 'O', 'CB']
                            if (rj, an) in atom_to_idx
                            and atom_to_idx[(rj, an)] > ib
                        ]
                        # Also include the chain-end atom if it's downstream
                        if end_idx > ib and end_idx not in downstream:
                            downstream.append(end_idx)
                        if downstream:
                            pivot_bonds.append((ia, ib, downstream))

                if pivot_bonds:
                    ccd_close(flat_pos, pivot_bonds, end_idx, target,
                              max_iter=ccd_iters, tol=ccd_tol)

            # Write CCD positions back into residues
            for (ri, aname), pos in zip(flat_atoms, flat_pos):
                residues[ri]['atoms'][aname] = pos

        # ── 4b. ProMod3 loop closing (replaces CCD when requested) ────────────
        if loop_closer == 'promod3':
            from alphatracer.utils.loop_closer import close_gap_promod3
            import gemmi as _gemmi
            import tempfile as _tempfile
            import os as _os

            # Write current structure to temp PDB, run ProMod3, read back
            with _tempfile.TemporaryDirectory() as _tmpdir:
                _tmp_in = _os.path.join(_tmpdir, "before_pm3.pdb")
                _tmp_out = _os.path.join(_tmpdir, "after_pm3.pdb")

                # Build a minimal gemmi structure from current residues
                _st = _gemmi.Structure()
                _model = _gemmi.Model('1')
                _chain = _gemmi.Chain(row.get('chain_id', 'A'))
                for ri, res_d in enumerate(residues):
                    _res = _gemmi.Residue()
                    _res.name = res_d.get('resname', 'ALA')
                    _res.seqid = _gemmi.SeqId(ri + 1, ' ')
                    for aname, xyz in res_d.get('atoms', {}).items():
                        _atom = _gemmi.Atom()
                        _atom.name = aname
                        _atom.pos = _gemmi.Position(*xyz)
                        _atom.element = _gemmi.Element(aname[0])
                        _res.add_atom(_atom)
                    _chain.add_residue(_res)
                _model.add_chain(_chain)
                _st.add_model(_model)
                _st.write_pdb(_tmp_in)

                _filled, _err = close_gap_promod3(
                    _st,
                    chain_id=row.get('chain_id', 'A'),
                    n_stem_seqid=None,
                    c_stem_seqid=None,
                    gap_seq=None,
                    target_seq=row['qseq'],
                    tpl_seq_with_gaps=row['sseq_alg'],
                )
                if _err is None and _filled is not None:
                    # Read back coordinates into residues dict
                    _filled_chain = _filled[0][row.get('chain_id', 'A')]
                    for _res in _filled_chain:
                        _ri = _res.seqid.num - 1
                        if 0 <= _ri < len(residues):
                            for _atom in _res:
                                residues[_ri]['atoms'][_atom.name] = [
                                    _atom.pos.x, _atom.pos.y, _atom.pos.z]

        # ── 5. OpenMM minimisation (skipped if no backbone clashes after CCD) ───
        system, top, pos_nm = build_openmm_system(residues)

        if _has_backbone_clash(residues):
            integrator = LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
            )
            sim = Simulation(top, system, integrator, platform=_get_platform())
            sim.context.setPositions(pos_nm)
            sim.minimizeEnergy(maxIterations=mm_iters)
            positions = sim.context.getState(getPositions=True).getPositions()
        else:
            positions = pos_nm

        with open(out_pdb, 'w') as f:
            PDBFile.writeFile(top, positions, f)

        return True, None, time.perf_counter() - t0

    except Exception as e:
        return False, str(e), time.perf_counter() - t0


# ── Download ──────────────────────────────────────────────────────────────────

def stage_download(afdb_ids, pdb_dir, threads):
    missing = [aid for aid in afdb_ids
               if not (os.path.exists(afdb_local_pdb(aid, pdb_dir))
                       and is_valid_pdb(afdb_local_pdb(aid, pdb_dir)))]
    print(f'  {len(afdb_ids)} accessions; {len(missing)} to download')
    if not missing:
        return
    fetch_afdb_pdbs(missing, pdb_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    indir  = args.input_dir.rstrip('/')
    pdb_dir = os.path.join(indir, 'AF_pdbs')
    outdir  = os.path.join(indir, 'output_pdbs_classB')
    os.makedirs(outdir, exist_ok=True)

    ca_only = not args.full_pdbs   # default is CA-only; --full-pdbs opts into full backbone

    print('=' * 60)
    if ca_only:
        print('AlphaTracer 1.0  —  Class B Pipeline  (CA-only trace)')
    else:
        print('AlphaTracer 1.0  —  Class B Pipeline  (CCD + backbone OpenMM)')
    print('=' * 60)
    print(f'  Input dir:      {indir}/')
    print(f'  Mode:           {"CA-only (fast)" if ca_only else "full backbone (CCD + OpenMM)"}')
    print(f'  Max indels:     {args.max_indels} (each ≤{args.max_indel_len} aa)  '
          f'| loop relaxed: {args.max_loop_indels} (each ≤{args.max_loop_indel_len} aa)')
    if not ca_only:
        print(f'  CCD:            {args.ccd_iters} iters, tol={args.ccd_tol} Å, '
              f'flank={args.flank}')
        print(f'  OpenMM:         {args.mm_iters} minimisation steps')
    print()

    # ── Load hits ─────────────────────────────────────────────────────────────
    global _HAS_DB_TYPE
    hits_df = None
    for fname in ['allhits.pq', 'tophits.pq']:
        p = os.path.join(indir, fname)
        if os.path.exists(p):
            hits_df = pl.read_parquet(p)
            print(f'Loaded {fname}: {len(hits_df)} rows')
            break
    if hits_df is None:
        sys.exit(f'No allhits.pq/tophits.pq in {indir}')
    _HAS_DB_TYPE = 'db_type' in hits_df.columns

    classA_path = os.path.join(indir, 'classA.pq')
    if not os.path.exists(classA_path):
        sys.exit(f'classA.pq not found in {indir}')
    classA_ids = set(pl.read_parquet(classA_path)['qseqid'].to_list())

    _sort_col = 'containment_value' if 'containment_value' in hits_df.columns else 'approx_pident'
    non_classA = hits_df
    if 'approx_pident' in hits_df.columns:
        non_classA = non_classA.filter(pl.col('approx_pident') >= args.min_pctsim)
    non_classA = (
        non_classA
        .filter(~pl.col('qseqid').is_in(classA_ids))
        .sort(_sort_col, descending=True)
        .group_by('qseqid', maintain_order=True)
        .agg(pl.all().first())
    )
    print(f'{len(classA_ids)} Class A  |  {len(non_classA)} non-Class A to check')

    # ── [1] NW alignment (parallel — parasail releases the GIL) ──────────────
    print(f'\n[1/4] Aligning {len(non_classA)} sequences (threads={args.threads})...')
    rows_list = list(non_classA.iter_rows(named=True))
    total     = len(rows_list)

    def _aln(row):
        return align_nw(row['qseqid'], row['sseqid'], row['full_qseq'], row['full_sseq'])

    aln_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for result in ex.map(_aln, rows_list):
            done += 1
            if result:
                aln_rows.append(result)
            if done % 200 == 0 or done == total:
                print(f'  {done}/{total}...', end='\r')
    print()

    if not aln_rows:
        print('No alignments. Exiting.'); return

    aln_df = pl.DataFrame(aln_rows,
                          schema=['qseqid','sseqid','qseq_alg','sseq_alg','alg_comp'],
                          orient='row')
    merged = non_classA.join(aln_df, on=['qseqid', 'sseqid'], how='inner')

    # Apply real NW identity threshold (approx_pident is kmer Jaccard, not NW)
    def _nw_pident(r):
        q, s = r['qseq_alg'], r['sseq_alg']
        start = len(q) - len(q.lstrip('-'))
        end   = len(q.rstrip('-')) or len(q)
        matches = sum(1 for a, b in zip(q[start:end], s[start:end]) if a == b and a != '-')
        aligned  = sum(1 for a, b in zip(q[start:end], s[start:end]) if a != '-' or b != '-')
        return 100.0 * matches / aligned if aligned else 0.0

    _thresh = args.min_pctsim
    pre_filter = len(merged)
    merged = merged.filter(
        pl.struct(['qseq_alg', 'sseq_alg']).map_elements(
            lambda r: _nw_pident(r) >= _thresh,
            return_dtype=pl.Boolean,
        )
    )
    print(f'  NW identity ≥{_thresh}%: {len(merged)}/{pre_filter} pass')

    # ── [2] Classify Class B ──────────────────────────────────────────────────
    print(f'\n[2/4] Classifying Class B '
          f'(strict: ≤{args.max_indels} indels ≤{args.max_indel_len} aa; '
          f'relaxed loop: ≤{args.max_loop_indels} indels ≤{args.max_loop_indel_len} aa)...')
    _mi, _mil, _mli, _mlil = (args.max_indels, args.max_indel_len,
                               args.max_loop_indels, args.max_loop_indel_len)
    classB_df = merged.filter(
        pl.struct(['qseq_alg', 'sseq_alg']).map_elements(
            lambda r: classify_classB(r['qseq_alg'], r['sseq_alg'],
                                      _mi, _mil, _mli, _mlil),
            return_dtype=pl.Boolean,
        )
    )
    classB_df.write_parquet(os.path.join(indir, 'classB.pq'))

    ins_counts = Counter()
    for row in classB_df.iter_rows(named=True):
        ni = len(re.findall(r'-+', row['sseq_alg']))
        nd = len(re.findall(r'-+', row['qseq_alg'].strip('-')))
        ins_counts[(ni, nd)] += 1
    print(f'  {len(classB_df)} / {len(non_classA)} qualify as Class B')
    for (ni, nd), c in sorted(ins_counts.items()):
        print(f'    {ni} insertion(s) + {nd} deletion(s): {c}')

    if len(classB_df) == 0:
        print('No Class B sequences.'); return

    if args.limit > 0:
        classB_df = classB_df.head(args.limit)
        print(f'  (limited to first {args.limit} sequences)')

    # ── [3] Download ──────────────────────────────────────────────────────────
    print('\n[3/4] Downloading reference PDBs...')
    rows_list_b = list(classB_df.iter_rows(named=True))
    if _HAS_DB_TYPE and 'db_type' in classB_df.columns:
        afdb_rows = [r for r in rows_list_b if (r.get('db_type') or 'afdb') == 'afdb']
        esm_rows  = [r for r in rows_list_b if (r.get('db_type') or 'afdb') == 'esm_atlas']
    else:
        afdb_rows = rows_list_b
        esm_rows  = []
    afdb_ids = {get_afdb_id(r['sseqid']) for r in afdb_rows if get_afdb_id(r['sseqid'])}
    if esm_rows:
        import threading as _threading
        print(f'  Pre-fetching {len(esm_rows)} ESM Atlas structure(s) (background)...')
        _esm_thread = _threading.Thread(
            target=fetch_esm_structures, args=(esm_rows, pdb_dir),
            kwargs={'n_workers': 16}, daemon=True)
        _esm_thread.start()
    else:
        _esm_thread = None
    stage_download(afdb_ids, pdb_dir, args.threads)
    if _esm_thread is not None:
        _esm_thread.join()

    # ── [4] Build structures (parallel — OpenMM releases the GIL) ────────────
    print(f'\n[4/4] Building {len(classB_df)} Class B structures '
          f'(threads={args.threads})...')
    n_ok = n_fail = 0
    ok_by_db: Counter = Counter()
    fail_by_db: Counter = Counter()
    timings = []
    failure_reasons = {}
    import threading as _threading, json as _json
    _counts_lock = _threading.Lock()
    _counts_path = os.path.join(indir, '.classB_db_counts')

    def _write_b_counts():
        tmp = _counts_path + '.tmp'
        with open(tmp, 'w') as f:
            _json.dump({'ok': dict(ok_by_db), 'fail': dict(fail_by_db)}, f)
        os.replace(tmp, _counts_path)

    def _build_one(row):
        qseqid  = row['qseqid']
        db_type = row.get('db_type', 'afdb') or 'afdb'
        out_pdb = os.path.join(outdir, f'classB:{qseqid}.pdb')
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            return qseqid, True, None, 0.0, True, db_type   # cached
        if ca_only:
            ok, err, elapsed = build_classB_ca_only(row, pdb_dir, out_pdb)
        else:
            ok, err, elapsed = build_classB_structure(
                row, pdb_dir, out_pdb,
                args.mm_iters, args.ccd_iters, args.ccd_tol, args.flank,
                loop_closer=args.loop_closer,
                max_indels=args.max_indels, max_indel_len=args.max_indel_len,
            )
        return qseqid, ok, err, elapsed, False, db_type

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(_build_one, row): row['qseqid']
                for row in classB_df.iter_rows(named=True)}
        for fut in as_completed(futs):
            qseqid, ok, err, elapsed, cached, db = fut.result()
            with _counts_lock:
                if cached:
                    n_ok += 1; ok_by_db[db] += 1
                elif ok:
                    print(f'  {qseqid}: OK  {elapsed:.2f}s')
                    n_ok += 1; ok_by_db[db] += 1; timings.append(elapsed)
                else:
                    print(f'  {qseqid}: FAILED — {err}')
                    failure_reasons[qseqid] = err or ''
                    n_fail += 1; fail_by_db[db] += 1
                _write_b_counts()

    # ── Mapping table ─────────────────────────────────────────────────────────
    table_rows = []
    for row in classB_df.iter_rows(named=True):
        qseqid  = row['qseqid']
        out_pdb = os.path.join(outdir, f'classB:{qseqid}.pdb')
        db_type = row.get('db_type', 'afdb') or 'afdb'
        if db_type == 'esm_atlas':
            protein_hash = row.get('protein_hash') or row['sseqid'].split('|')[0]
            ref_pdb = _esm_local_pdb(protein_hash, pdb_dir)
        else:
            afdb_id = get_afdb_id(row['sseqid'])
            ref_pdb = afdb_local_pdb(afdb_id, pdb_dir) if afdb_id else ''
        ok      = os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0
        table_rows.append({
            'query_id':  qseqid,
            'built_pdb': out_pdb,
            'ref_pdb':   ref_pdb,
            'status':    'ok' if ok else 'failed',
            'reason':    '' if ok else failure_reasons.get(qseqid, 'skipped'),
        })

    table_df = pl.DataFrame(table_rows)
    table_path = os.path.join(indir, 'classB_pdb_table.tsv')
    table_df.write_csv(table_path, separator='\t')

    # Print table
    col_w = [
        max(len('query_id'),   max(len(r['query_id'])  for r in table_rows)),
        max(len('built_pdb'),  max(len(r['built_pdb']) for r in table_rows)),
        max(len('ref_pdb'),    max(len(r['ref_pdb'])   for r in table_rows)),
        max(len('status'),     max(len(r['status'])    for r in table_rows)),
        max(len('reason'),     max(len(r['reason'])    for r in table_rows)),
    ]
    sep = '  '.join('-' * w for w in col_w)
    hdr = '  '.join(h.ljust(w) for h, w in zip(
        ['query_id', 'built_pdb', 'ref_pdb', 'status', 'reason'], col_w))

    print()
    print(hdr)
    print(sep)
    for r in table_rows:
        print('  '.join(v.ljust(w) for v, w in zip(
            [r['query_id'], r['built_pdb'], r['ref_pdb'], r['status'], r['reason']],
            col_w)))

    with open(os.path.join(indir, '.classB_total'), 'w') as _f:
        _f.write(str(n_ok))

    print()
    print('=' * 60)
    print('Class B complete.')
    print(f'  Class B sequences:  {len(classB_df)}')
    print(f'  PDBs written:       {n_ok}')
    if _HAS_DB_TYPE:
        print(f'    AFDB:           {ok_by_db["afdb"]}')
        print(f'    ESM Atlas:      {ok_by_db["esm_atlas"]}')
    print(f'  PDBs failed:        {n_fail}')
    if _HAS_DB_TYPE and n_fail:
        print(f'    AFDB:           {fail_by_db["afdb"]}')
        print(f'    ESM Atlas:      {fail_by_db["esm_atlas"]}')
    if timings:
        print(f'  Time per structure: min={min(timings):.2f}s  '
              f'mean={sum(timings)/len(timings):.2f}s  '
              f'max={max(timings):.2f}s')
    print(f'  Table:  {table_path}')
    print(f'  Output: {outdir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
