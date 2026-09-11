"""Loop closing backend: CCD (default) or ProMod3 (fragment DB + CCD).

Usage
-----
from loop_closer import close_gap, BACKEND_CCD, BACKEND_PROMOD3

# CCD (no extra dependencies)
coords = close_gap(positions, pivot_bonds, end_idx, target,
                   backend=BACKEND_CCD)

# ProMod3 (requires OST + ProMod3 in the running Python environment)
coords = close_gap_promod3(structure_path, gap_seq, n_stem_resnum,
                            c_stem_resnum, chain_id)

ProMod3 integration strategy
-----------------------------
AlphaTracer uses gemmi for structure I/O; ProMod3 uses OpenStructure (OST).
The bridge is a temporary PDB file:
  gemmi structure → temp PDB → OST EntityHandle → ProMod3 → temp PDB → gemmi

This avoids any runtime coupling between the two libraries.
"""
import os
import tempfile
import threading
import time
import numpy as np
from pathlib import Path

BACKEND_CCD     = "ccd"
BACKEND_PROMOD3 = "promod3"
BACKEND_KIC     = "kic"

# ── Lazy ProMod3 globals (loaded once per process) ────────────────────────────
_pm3_loaded    = False
_frag_db       = None
_structure_db  = None
_torsion_sampler = None

# ── Lazy native KIC globals ───────────────────────────────────────────────────
_native_frag_db = None
_native_lock    = threading.Lock()


def _load_native():
    """Load the native fragment DB (once per process, thread-safe)."""
    global _native_frag_db
    if _native_frag_db is not None:
        return
    with _native_lock:
        if _native_frag_db is not None:  # double-checked locking
            return
        from alphatracer.utils.frag_db import load as _load_fdb
        data_dir = Path(__file__).parent.parent / 'data'
        for name in ('fragdb_local.npz', 'fragdb.npz'):
            p = data_dir / name
            if p.exists():
                _native_frag_db = _load_fdb(str(p))
                return
        raise FileNotFoundError(
            f"Native fragment DB not found in {data_dir}. "
            "Run scripts/setup/dump_fragdb.py to generate it."
        )


def _load_promod3():
    """Import OST/ProMod3 and load databases (once per process)."""
    global _pm3_loaded, _frag_db, _structure_db, _torsion_sampler
    if _pm3_loaded:
        return
    try:
        from promod3 import loop as pm3_loop
    except ImportError as e:
        raise RuntimeError(
            "ProMod3 not importable. Run inside the ProMod3 Docker container "
            "or install OST + ProMod3. Original error: " + str(e))
    _frag_db         = pm3_loop.LoadFragDB()
    _structure_db    = pm3_loop.LoadStructureDB()
    _torsion_sampler = pm3_loop.LoadTorsionSamplerCoil()
    _pm3_loaded = True


# ── CCD (existing AlphaTracer implementation, unchanged) ──────────────────────

def _rotation_matrix(axis, angle):
    """Rodrigues rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _ccd_step(positions, pivot_a, pivot_b, downstream_indices, end_idx, target):
    """Single CCD step: rotate downstream atoms around pivot_a→pivot_b."""
    axis   = positions[pivot_b] - positions[pivot_a]
    norm   = np.linalg.norm(axis)
    if norm < 1e-8:
        return
    axis /= norm
    r_end = positions[end_idx]  - positions[pivot_a]
    r_tgt = target              - positions[pivot_a]
    r_end -= np.dot(r_end, axis) * axis
    r_tgt -= np.dot(r_tgt, axis) * axis
    n_end, n_tgt = np.linalg.norm(r_end), np.linalg.norm(r_tgt)
    if n_end < 1e-8 or n_tgt < 1e-8:
        return
    r_end /= n_end
    r_tgt /= n_tgt
    cos_a = np.clip(np.dot(r_end, r_tgt), -1.0, 1.0)
    cross = np.cross(r_end, r_tgt)
    sin_a = np.dot(cross, axis)
    angle = np.arctan2(sin_a, cos_a)
    R = _rotation_matrix(axis, angle)
    for idx in downstream_indices:
        positions[idx] = positions[pivot_a] + R @ (positions[idx] - positions[pivot_a])


def ccd_close(positions, pivot_bonds, end_idx, target, max_iter=200, tol=0.15):
    """
    Cyclic Coordinate Descent loop closure (original AlphaTracer implementation).

    Parameters
    ----------
    positions   : (N, 3) float array — modified in place
    pivot_bonds : list of (a_idx, b_idx, downstream_indices)
    end_idx     : index of atom that must reach target
    target      : (3,) target position
    max_iter    : iteration limit
    tol         : convergence threshold (Å)
    """
    for _ in range(max_iter):
        if np.linalg.norm(positions[end_idx] - target) < tol:
            break
        for pivot_a, pivot_b, downstream in pivot_bonds:
            _ccd_step(positions, pivot_a, pivot_b, downstream, end_idx, target)
    return positions


# ── ProMod3 loop closer ───────────────────────────────────────────────────────

def close_gap_promod3(gemmi_st, chain_id, n_stem_seqid, c_stem_seqid,
                      gap_seq, target_seq, tpl_seq_with_gaps):
    """
    Close a single gap in a gemmi Structure using ProMod3.

    Writes the structure to a temp PDB, loads into OST, builds a
    ModellingHandle with the gap encoded in the alignment, runs
    FillLoopsByDatabase (falling back to FillLoopsByMonteCarlo if needed),
    then writes back and re-reads with gemmi.

    Parameters
    ----------
    gemmi_st          : gemmi.Structure — full structure (will be written to temp PDB)
    chain_id          : str — chain identifier
    n_stem_seqid      : int — 1-based sequence number of residue BEFORE gap
    c_stem_seqid      : int — 1-based sequence number of residue AFTER gap
    gap_seq           : str — amino acid sequence to insert (single-letter)
    target_seq        : str — full target sequence (no gaps)
    tpl_seq_with_gaps : str — aligned template sequence (with '-' for gap)

    Returns
    -------
    gemmi.Structure with the gap filled, or None if ProMod3 failed.
    """
    import gemmi
    from promod3 import modelling, loop as pm3_loop
    from ost import io, seq as ost_seq

    _load_promod3()

    with tempfile.TemporaryDirectory() as tmpdir:
        tpl_pdb = os.path.join(tmpdir, "template.pdb")
        out_pdb = os.path.join(tmpdir, "filled.pdb")

        # Write template to PDB
        gemmi_st.write_pdb(tpl_pdb)

        # Load in OST
        tpl_ost = io.LoadPDB(tpl_pdb)

        # Build alignment
        aln = ost_seq.CreateAlignment(
            ost_seq.CreateSequence('trg', target_seq),
            ost_seq.CreateSequence('tpl', tpl_seq_with_gaps))
        aln.AttachView(1, tpl_ost.CreateFullView())

        try:
            mhandle = modelling.BuildRawModel(aln)
        except Exception as e:
            return None, f"BuildRawModel failed: {e}"

        if len(mhandle.gaps) == 0:
            return gemmi_st, None  # nothing to close

        t0 = time.perf_counter()

        # Try fragment DB first
        modelling.FillLoopsByDatabase(
            mhandle, _frag_db, _structure_db, _torsion_sampler)

        # Fall back to Monte Carlo for any remaining gaps
        if len(mhandle.gaps) > 0:
            modelling.FillLoopsByMonteCarlo(
                mhandle, _torsion_sampler, mc_steps=5000)

        elapsed = time.perf_counter() - t0
        gaps_remaining = len(mhandle.gaps)

        # Write filled model
        io.SavePDB(mhandle.model, out_pdb)

        # Re-read with gemmi
        filled_st = gemmi.read_structure(out_pdb)

    return filled_st, None if gaps_remaining == 0 else f"{gaps_remaining} gap(s) unclosed"


# ── Native KIC loop closer ────────────────────────────────────────────────────

def close_gap_kic(residues: list, gap_sites: list) -> list:
    """
    Close insertion gaps using the native fragment DB + analytical KIC.
    No Docker or ProMod3 required.

    For each insertion gap_site, queries the fragment DB for candidates
    matching the stem-pair geometry, then applies analytical KIC to close
    each candidate to the C-terminal stem. The first successfully closed
    fragment replaces the CCD-placed residue backbone in `residues`.

    Parameters
    ----------
    residues  : list of residue dicts (modified in place)
    gap_sites : list of (gap_before_ri, gap_type, loop_len) 3-tuples

    Returns
    -------
    List of (gap_before_ri, closed: bool) for each insertion gap attempted.
    """
    from alphatracer.utils.kic import close_loop as _kic_close

    _load_native()
    db = _native_frag_db

    results = []
    for (gap_before_ri, gap_type, loop_len) in gap_sites:
        if gap_type != 'insertion' or loop_len < 3:
            continue

        n_stem_ri = gap_before_ri
        loop_start = gap_before_ri + 1
        loop_end   = gap_before_ri + loop_len      # inclusive
        c_stem_ri  = gap_before_ri + loop_len + 1

        if c_stem_ri >= len(residues):
            results.append((gap_before_ri, False))
            continue

        n_atoms = residues[n_stem_ri]['atoms']
        c_atoms = residues[c_stem_ri]['atoms']
        if not all(k in n_atoms for k in ('N', 'CA', 'C')):
            results.append((gap_before_ri, False))
            continue
        if not all(k in c_atoms for k in ('N', 'CA', 'C')):
            results.append((gap_before_ri, False))
            continue

        n_stem_N  = np.array(n_atoms['N'],  float)
        n_stem_CA = np.array(n_atoms['CA'], float)
        n_stem_C  = np.array(n_atoms['C'],  float)
        c_stem_N  = np.array(c_atoms['N'],  float)
        c_stem_CA = np.array(c_atoms['CA'], float)
        c_stem_C  = np.array(c_atoms['C'],  float)

        # Query fragment DB — candidates already in absolute coords at n_stem
        frag_coords, _ = db.query(
            n_stem_N, n_stem_CA, n_stem_C,
            c_stem_N, c_stem_CA, c_stem_C,
            frag_len=loop_len, extra_bins=1,
        )

        if len(frag_coords) == 0:
            results.append((gap_before_ri, False))
            continue

        # Try candidates through analytical KIC until one closes
        closed = None
        for frag in frag_coords:
            sol = _kic_close(
                frag,
                n_stem_N, n_stem_CA, n_stem_C,
                c_stem_N, c_stem_CA, c_stem_C,
            )
            if sol is not None:
                closed = sol
                break

        if closed is None:
            results.append((gap_before_ri, False))
            continue

        # Write closed backbone into residues
        ATOM_NAMES = ('N', 'CA', 'C', 'O')
        for j, ri in enumerate(range(loop_start, loop_end + 1)):
            for k, aname in enumerate(ATOM_NAMES):
                residues[ri]['atoms'][aname] = closed[j, k].tolist()

        results.append((gap_before_ri, True))

    return results


# ── Public interface ──────────────────────────────────────────────────────────

def get_available_backends():
    """Return list of backends available in the current environment."""
    backends = [BACKEND_CCD]
    try:
        import promod3  # noqa: F401
        import ost      # noqa: F401
        backends.append(BACKEND_PROMOD3)
    except ImportError:
        pass
    try:
        _load_native()
        backends.append(BACKEND_KIC)
    except (FileNotFoundError, Exception):
        pass
    return backends


def promod3_available():
    try:
        import promod3  # noqa: F401
        import ost      # noqa: F401
        return True
    except ImportError:
        return False
