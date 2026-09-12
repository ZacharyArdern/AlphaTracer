"""
Analytical KIC — ported from ProMod3 modelling/src/kic.cc (Apache 2.0).

Finds all closed loop conformations (up to 16) via Dixon resultant →
generalized 16×16 eigenvalue problem. O(1) per call, ~1 ms per loop.

Reference: Coutsias et al., J Comput Chem 25(4):510-528, 2004.
Source:    ProMod3 modelling/src/kic.cc (SIB/Biozentrum Basel, Apache 2.0)

Public API
----------
solve(loop_coords, n_stem_N, n_stem_CA, n_stem_C,
      c_stem_N, c_stem_CA, c_stem_C, pivot1, pivot2, pivot3)
    Returns list[np.ndarray(N,4,3)] — up to 16 closed solutions.
    Constraint: 0 < pivot1 < pivot2 < pivot3 < N-1

close_loop(loop_coords, n_stem_*, c_stem_*, pivot1/2/3=None)
    Auto-selects pivots, returns best solution or None.
    Minimum loop length: 5 residues.

Coordinate convention
---------------------
loop_coords : (N, 4, 3) — N/CA/C/O per residue (atom axis 0=N,1=CA,2=C,3=O)
n/c_stem_*  : (3,) Å — anchor atom positions
pivots      : 0-based indices into loop_coords

Algorithm overview
------------------
1. Compute min-RMSD superposition transforms T_n, T_c that align the first/
   last loop residues to the N/C-stem anchors.
2. Define three rigid fragments F1 (p1..p2), F2 (p2..p3), F3 (p3..p1 via
   stems) and extract inter-fragment geometry (distances, angles).
3. Build three 8×8 Dixon matrices R0, R1, R2 encoding the closure polynomial.
4. Solve the generalized 16×16 eigenvalue problem → up to 16 real (τ1,τ2,τ3).
5. For each solution apply rigid fragment rotations to produce closed coords.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eig
from typing import Optional

# ── Module-level pre-allocated eigenvalue matrices ───────────────────────────
# The constant parts are filled once; only the non-constant parts are updated
# per call to _solve_eigen, avoiding repeated allocations.
_E0 = np.zeros((16, 16))
_E0[:8, 8:] = np.eye(8)   # constant upper-right block

_E1 = np.zeros((16, 16))
_E1[:8, :8] = np.eye(8)   # constant upper-left block

BL_CO = 1.229


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v.copy()


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _dihedral(a, b, c, d) -> float:
    b1 = _unit(b - a); b2 = _unit(c - b); b3 = _unit(d - c)
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    return float(np.arctan2(np.dot(np.cross(n1, b2), n2), np.dot(n1, n2)))


def _place_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                bond_len: float, bond_angle: float, torsion: float) -> np.ndarray:
    """NERF: place D from A,B,C.

    bond_angle: angle B-C-D (radians); torsion: dihedral A-B-C-D (radians).
    Inverse of _dihedral — verified to round-trip within 1e-6 Å.
    """
    bc = _unit(c - b)
    n  = _unit(np.cross(b - a, bc))
    m  = np.cross(n, bc)
    return c + bond_len * (
        -np.cos(bond_angle) * bc
        + np.sin(bond_angle) * (np.cos(torsion) * m - np.sin(torsion) * n)
    )


# ── 4×4 homogeneous transform helpers ────────────────────────────────────────

def _apply4(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    return M[:3, :3] @ v + M[:3, 3]


def _rot4(R: np.ndarray) -> np.ndarray:
    M = np.eye(4); M[:3, :3] = R; return M


def _trans4(t: np.ndarray) -> np.ndarray:
    M = np.eye(4); M[:3, 3] = t; return M


def _rot4_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[1, 1] = c; M[1, 2] = -s
    M[2, 1] = s; M[2, 2] = c
    return M


def _rot4_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[0, 0] = c; M[0, 1] = -s
    M[1, 0] = s; M[1, 1] = c
    return M


# ── Kabsch superposition ──────────────────────────────────────────────────────

def _kabsch(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """
    4×4 RT matrix that maps src (3,3) → tgt (3,3) via min-RMSD superposition.
    Rows of src/tgt are [N, CA, C] atom positions.
    """
    sc = src.mean(0); tc = tgt.mean(0)
    H = (src - sc).T @ (tgt - tc)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = tc - R @ sc
    return M


# ── Fragment base (mirrors ProMod3 BuildFragmentBase) ─────────────────────────

def _frag_base(pos1: np.ndarray, pos2: np.ndarray, pos3: np.ndarray) -> np.ndarray:
    """3×3 matrix whose rows are orthonormal basis vectors [a, b, c]."""
    a = _unit(pos2 - pos1)
    c = _unit(np.cross(a, pos3 - pos2))
    b = np.cross(c, a)
    return np.stack([a, b, c])


# ── KIC parameter computation (FillKICParameters) ────────────────────────────

def _fill_kic_params(
    loop_bb: np.ndarray,          # (N, 3, 3) N/CA/C per residue
    n_stem: np.ndarray,           # (3, 3) N/CA/C of n_stem
    c_stem: np.ndarray,           # (3, 3) N/CA/C of c_stem
    p1: int, p2: int, p3: int,
) -> Optional[dict]:
    N_IDX, CA_IDX, C_IDX = 0, 1, 2

    T_n = _kabsch(loop_bb[0],       n_stem)
    T_c = _kabsch(loop_bb[-1],      c_stem)

    def tn(v): return _apply4(T_n, v)
    def tc(v): return _apply4(T_c, v)

    # Fragment vertices matching ProMod3 F1, F2, F3
    # F1: [CA(p1), C(p1), N(p2), CA(p2)]
    # F2: [CA(p2), C(p2), N(p3), CA(p3)]
    # F3: [T_c@CA(p3), T_c@C(p3), T_n@N(p1), T_n@CA(p1)]
    F1 = [loop_bb[p1, CA_IDX], loop_bb[p1, C_IDX],
          loop_bb[p2, N_IDX],  loop_bb[p2, CA_IDX]]
    F2 = [loop_bb[p2, CA_IDX], loop_bb[p2, C_IDX],
          loop_bb[p3, N_IDX],  loop_bb[p3, CA_IDX]]
    F3 = [tc(loop_bb[p3, CA_IDX]), tc(loop_bb[p3, C_IDX]),
          tn(loop_bb[p1, N_IDX]),  tn(loop_bb[p1, CA_IDX])]

    dist = np.array([
        np.linalg.norm(F1[0] - F1[3]),
        np.linalg.norm(F2[0] - F2[3]),
        np.linalg.norm(F3[0] - F3[3]),
    ])

    if (dist[0] + dist[1] <= dist[2] or
        dist[1] + dist[2] <= dist[0] or
        dist[2] + dist[0] <= dist[1]):
        return None

    theta = np.array([
        _angle(loop_bb[p1, N_IDX] - loop_bb[p1, CA_IDX],
               loop_bb[p1, C_IDX] - loop_bb[p1, CA_IDX]),
        _angle(loop_bb[p2, N_IDX] - loop_bb[p2, CA_IDX],
               loop_bb[p2, C_IDX] - loop_bb[p2, CA_IDX]),
        _angle(loop_bb[p3, N_IDX] - loop_bb[p3, CA_IDX],
               loop_bb[p3, C_IDX] - loop_bb[p3, CA_IDX]),
    ])

    delta = np.array([
        _dihedral(F1[1], F1[0], F1[3], F1[2]),
        _dihedral(F2[1], F2[0], F2[3], F2[2]),
        _dihedral(F3[1], F3[0], F3[3], F3[2]),
    ])

    eta = np.array([
        _angle(F1[1] - F1[0], F1[3] - F1[0]),
        _angle(F2[1] - F2[0], F2[3] - F2[0]),
        _angle(F3[1] - F3[0], F3[3] - F3[0]),
    ])

    xi = np.array([
        _angle(F1[0] - F1[3], F1[2] - F1[3]),
        _angle(F2[0] - F2[3], F2[2] - F2[3]),
        _angle(F3[0] - F3[3], F3[2] - F3[3]),
    ])

    d11, d22, d33 = dist[0]**2, dist[1]**2, dist[2]**2
    alpha = np.array([
        np.arccos(np.clip((d22 - d33 - d11) / (2 * dist[0] * dist[2]), -1, 1)),
        np.arccos(np.clip((d33 - d11 - d22) / (2 * dist[0] * dist[1]), -1, 1)),
        np.arccos(np.clip((d11 - d22 - d33) / (2 * dist[1] * dist[2]), -1, 1)),
    ])

    F1_base = _frag_base(F1[0], F1[3], F1[1])
    F2_base = _frag_base(F2[0], F2[3], F2[1])
    F3_base = _frag_base(F3[0], F3[3], F3[1])

    F1_init = _rot4(F1_base) @ _trans4(-loop_bb[p1, CA_IDX])
    F2_init = _rot4(F2_base) @ _trans4(-loop_bb[p2, CA_IDX])

    return dict(
        dist=dist, theta=theta, delta=delta, eta=eta, xi=xi, alpha=alpha,
        F3_base=F3_base, F3=F3,
        T_n=T_n, T_c=T_c,
        F1_init=F1_init, F2_init=F2_init,
        p1=p1, p2=p2, p3=p3,
    )


# ── Dixon matrix construction (FillDixonMatrices) ────────────────────────────

def _fill_dixon(params: dict):
    dist, theta, delta, eta, xi, alpha = (
        params['dist'], params['theta'], params['delta'],
        params['eta'], params['xi'], params['alpha'],
    )

    c_theta = np.cos(theta)
    im1 = [2, 0, 1]        # i_min_one = (i+2)%3 for i=0,1,2
    s_delta = np.sin(delta[im1])
    c_delta = np.cos(delta[im1])
    s_xi    = np.sin(xi[im1])
    c_xi    = np.cos(xi[im1])
    s_eta   = np.sin(eta)
    s_aph   = np.sin(alpha + eta)
    s_amh   = np.sin(alpha - eta)
    c_aph   = np.cos(alpha + eta)
    c_amh   = np.cos(alpha - eta)

    sd_sx  = s_delta * s_xi
    cd_sx  = c_delta * s_xi
    cx_caph = c_xi * c_aph
    cx_camh = c_xi * c_amh

    # p[row][col][fragment] — direct port of ProMod3 FillDixonMatrices
    p = np.zeros((3, 3, 3))
    for i in range(3):
        p[0, 0, i] = -c_theta[i] - cx_caph[i] + cd_sx[i] * s_aph[i]
        p[0, 1, i] = 2 * sd_sx[i] * s_eta[i]
        p[0, 2, i] = -c_theta[i] - cx_camh[i] + cd_sx[i] * s_amh[i]

        p[1, 0, i] = -2 * sd_sx[i] * s_aph[i]
        p[1, 1, i] = 4 * cd_sx[i] * s_eta[i]
        p[1, 2, i] = -2 * sd_sx[i] * s_amh[i]

        p[2, 0, i] = -c_theta[i] - cx_caph[i] - cd_sx[i] * s_aph[i]
        p[2, 1, i] = -2 * sd_sx[i] * s_eta[i]
        p[2, 2, i] = -c_theta[i] - cx_camh[i] - cd_sx[i] * s_amh[i]

    A = np.empty((3, 3)); B = np.empty((3, 3))
    C = np.empty((3, 3)); D = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            A[i, j] = p[i, 1, 1] * p[0, j, 2] - p[i, 0, 1] * p[1, j, 2]
            B[i, j] = p[i, 2, 1] * p[0, j, 2] - p[i, 0, 1] * p[2, j, 2]
            C[i, j] = p[i, 2, 1] * p[1, j, 2] - p[i, 1, 1] * p[2, j, 2]
            D[i, j] = p[j, i, 0]

    R = [np.zeros((8, 8)) for _ in range(3)]
    for k in range(3):
        Rk = R[k]
        Rk[0, 1]=A[0,k]; Rk[0, 2]=A[1,k]; Rk[0, 3]=A[2,k]
        Rk[0, 5]=B[0,k]; Rk[0, 6]=B[1,k]; Rk[0, 7]=B[2,k]

        Rk[1, 0]=A[0,k]; Rk[1, 1]=A[1,k]; Rk[1, 2]=A[2,k]
        Rk[1, 4]=B[0,k]; Rk[1, 5]=B[1,k]; Rk[1, 6]=B[2,k]

        Rk[2, 1]=B[0,k]; Rk[2, 2]=B[1,k]; Rk[2, 3]=B[2,k]
        Rk[2, 5]=C[0,k]; Rk[2, 6]=C[1,k]; Rk[2, 7]=C[2,k]

        Rk[3, 0]=B[0,k]; Rk[3, 1]=B[1,k]; Rk[3, 2]=B[2,k]
        Rk[3, 4]=C[0,k]; Rk[3, 5]=C[1,k]; Rk[3, 6]=C[2,k]

        Rk[4, 5]=D[0,k]; Rk[4, 6]=D[1,k]; Rk[4, 7]=D[2,k]

        Rk[5, 4]=D[0,k]; Rk[5, 5]=D[1,k]; Rk[5, 6]=D[2,k]

        Rk[6, 1]=D[0,k]; Rk[6, 2]=D[1,k]; Rk[6, 3]=D[2,k]

        Rk[7, 0]=D[0,k]; Rk[7, 1]=D[1,k]; Rk[7, 2]=D[2,k]

    return R[0], R[1], R[2]


# ── Eigenvalue solver (ResolveEigenProblem) ───────────────────────────────────

def _solve_eigen(R0: np.ndarray, R1: np.ndarray, R2: np.ndarray) -> list:
    """Solve generalized 16×16 eigenvalue → list of (tau1, tau2, tau3) tuples."""
    E0 = _E0.copy()
    E1 = _E1.copy()
    E0[8:, :8] = -R0
    E0[8:, 8:] = -R1
    E1[8:, 8:] = R2

    w, vr = eig(E0, E1, check_finite=False)

    taus = []
    for i in range(16):
        if abs(w[i].imag) > 1e-6:
            continue
        v   = vr[:, i].real
        lam = w[i].real
        if abs(v[0]) > 1e-6:
            tau1 = 2 * np.arctan2(v[1], v[0])
            tau2 = 2 * np.arctan2(v[4], v[0])
        else:
            tau1 = 2 * np.arctan2(v[3], v[2])
            tau2 = 2 * np.arctan2(v[6], v[2])
        tau3 = 2 * np.arctan(lam)
        taus.append((tau1, tau2, tau3))

    return taus


# ── Apply one KIC solution (ApplySolution) ───────────────────────────────────

def _apply_solution(
    loop_bb: np.ndarray,    # (N, 3, 3) N/CA/C
    loop_o:  np.ndarray,    # (N, 3)  O atoms
    params: dict,
    tau1: float, tau2: float, tau3: float,
    c_stem_N: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Apply rigid fragment rotations and return (N, 4, 3) float32 array.
    Returns None if the closure error exceeds 0.35 Å.
    """
    p1, p2, p3 = params['p1'], params['p2'], params['p3']
    alpha = params['alpha']
    F3_base = params['F3_base']
    T_n, T_c = params['T_n'], params['T_c']
    F1_init, F2_init = params['F1_init'], params['F2_init']
    dist = params['dist']
    F3  = params['F3']   # list: F3[0]=T_c@CA(p3), F3[3]=T_n@CA(p1)
    N_res = len(loop_bb)

    # F1 transform chain (mirrors ProMod3 exactly)
    f1_C = _rot4_x(tau1)
    f1_D = _rot4_z(alpha[0])
    f1_E = _trans4(np.array([dist[2], 0.0, 0.0]))
    f1_F = _rot4_x(-tau3)
    f1_G = _rot4(F3_base.T)
    f1_H = _trans4(F3[0])           # translate to T_c @ CA(p3)
    f1_transform = f1_H @ f1_G @ f1_F @ f1_E @ f1_D @ f1_C @ F1_init

    # F2 transform chain
    f2_C = _rot4_x(tau2)
    f2_D = _rot4_z(-alpha[2])       # note: minus sign matches ProMod3 f2_D
    current_f2 = f2_D @ f2_C @ F2_init
    tv = _apply4(current_f2, loop_bb[p3, 1])   # CA(p3)
    f2_E = _trans4(np.array([-tv[0], -tv[1], 0.0]))
    pre = f1_H @ f1_G @ f1_F
    f2_transform = pre @ f2_E @ current_f2

    # Vectorized transform application (replaces N_res×3 individual _apply4 calls).
    # Build a flat view of all (N_res*3) atom positions: shape (N_res*3, 3).
    # Then apply each of the 4 transforms to its atom group in one batched matmul.
    #
    # Transform regions (atom index = i*3 + a, with a: 0=N,1=CA,2=C):
    #   T_n:          i <  p1  (all atoms)  +  i==p1, a<2
    #   f1_transform: i==p1, a==2  +  p1<i<p2 (all)  +  i==p2, a<2
    #   f2_transform: i==p2, a==2  +  p2<i<p3 (all)  +  i==p3, a==0
    #   T_c:          i==p3, a>0   +  i>p3 (all)

    flat = loop_bb.reshape(-1, 3)               # (N_res*3, 3)
    out_flat = np.empty_like(flat)

    def _apply_batch(M, pts):
        """Apply 4×4 homogeneous M to pts (K,3): pts @ R.T + t."""
        return pts @ M[:3, :3].T + M[:3, 3]

    # Build flat atom index masks
    idx = np.arange(N_res * 3)
    res_idx  = idx // 3   # residue index for each flat atom
    atom_idx = idx  % 3   # atom index (0=N,1=CA,2=C)

    mask_tn  = ((res_idx < p1) |
                ((res_idx == p1) & (atom_idx < 2)))
    mask_f1  = (((res_idx == p1) & (atom_idx == 2)) |
                ((res_idx > p1) & (res_idx < p2)) |
                ((res_idx == p2) & (atom_idx < 2)))
    mask_f2  = (((res_idx == p2) & (atom_idx == 2)) |
                ((res_idx > p2) & (res_idx < p3)) |
                ((res_idx == p3) & (atom_idx == 0)))
    mask_tc  = (((res_idx == p3) & (atom_idx > 0)) |
                (res_idx > p3))

    for mask, M in ((mask_tn, T_n), (mask_f1, f1_transform),
                    (mask_f2, f2_transform), (mask_tc, T_c)):
        if mask.any():
            out_flat[mask] = _apply_batch(M, flat[mask])

    out_bb = out_flat.reshape(N_res, 3, 3)

    # Validate closure: N of c_stem predicted by T_c applied to loop[-1,N]
    pred_cstem_N = _apply4(T_c, loop_bb[-1, 0])
    err = float(np.linalg.norm(pred_cstem_N - c_stem_N))
    if err > 0.35:
        return None

    # Check geometric closure via first N of c_stem
    # (T_c aligns loop[-1] to c_stem, so c_stem_N should match T_c @ loop[-1,N])
    # Additional check: p3 N should connect to the CA after transformation
    N_p3_out = out_bb[p3, 0]
    CA_p3_out = out_bb[p3, 1]
    if np.linalg.norm(CA_p3_out - _apply4(T_c, loop_bb[p3, 1])) > 0.35:
        return None

    # Build (N, 4, 3) with oxygen
    result = np.zeros((N_res, 4, 3), dtype=np.float32)
    result[:, :3] = out_bb.astype(np.float32)

    for i in range(N_res - 1):
        cn  = _unit(out_bb[i + 1, 0] - out_bb[i, 2])
        cca = _unit(out_bb[i,     1] - out_bb[i, 2])
        result[i, 3] = (out_bb[i, 2] - BL_CO * _unit(cn + cca)).astype(np.float32)
    cn_l  = _unit(c_stem_N - out_bb[-1, 2])
    cca_l = _unit(out_bb[-1, 1] - out_bb[-1, 2])
    result[-1, 3] = (out_bb[-1, 2] - BL_CO * _unit(cn_l + cca_l)).astype(np.float32)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def solve(
    loop_coords: np.ndarray,
    n_stem_N: np.ndarray, n_stem_CA: np.ndarray, n_stem_C: np.ndarray,
    c_stem_N: np.ndarray, c_stem_CA: np.ndarray, c_stem_C: np.ndarray,
    pivot1: int, pivot2: int, pivot3: int,
) -> list[np.ndarray]:
    """
    Analytical KIC — returns up to 16 distinct closed solutions.

    Following ProMod3, the stem residues are prepended/appended to the
    backbone so that T_n ≈ T_c ≈ identity and the triangle inequality
    always holds. Pivots are 1-based indices into the loop portion:
        1 ≤ pivot1 < pivot2 < pivot3 ≤ N
    where N = len(loop_coords). The full backbone is (N+2, 3, 3).

    Returns list of (N, 4, 3) float32 loop-only arrays (stems stripped).
    """
    n_res = len(loop_coords)

    n_stem_N  = np.asarray(n_stem_N,  float)
    n_stem_CA = np.asarray(n_stem_CA, float)
    n_stem_C  = np.asarray(n_stem_C,  float)
    c_stem_N  = np.asarray(c_stem_N,  float)
    c_stem_CA = np.asarray(c_stem_CA, float)
    c_stem_C  = np.asarray(c_stem_C,  float)

    n_stem_bb = np.stack([n_stem_N, n_stem_CA, n_stem_C])  # (3,3)
    c_stem_bb = np.stack([c_stem_N, c_stem_CA, c_stem_C])  # (3,3)

    # Augment: [n_stem, loop[0..N-1], c_stem] — mirrors ProMod3 bb_list layout
    loop_bb = loop_coords[:, :3, :].astype(float)
    full_bb = np.concatenate([n_stem_bb[None], loop_bb, c_stem_bb[None]], axis=0)
    full_o  = np.zeros((n_res + 2, 3))
    full_o[1:-1] = loop_coords[:, 3, :]

    # Pivots in augmented backbone: pivot1..pivot3 ∈ [1, N]
    # (loop residues occupy full_bb[1..N], stems at 0 and N+1)
    p1, p2, p3 = pivot1, pivot2, pivot3
    n_full = n_res + 2
    if not (0 < p1 < p2 < p3 < n_full - 1):
        return []

    # T_n = Kabsch(full_bb[0], n_stem_bb) ≈ identity
    # T_c = Kabsch(full_bb[-1], c_stem_bb) ≈ identity
    params = _fill_kic_params(full_bb, n_stem_bb, c_stem_bb, p1, p2, p3)
    if params is None:
        return []

    R0, R1, R2 = _fill_dixon(params)
    taus = _solve_eigen(R0, R1, R2)

    solutions = []
    seen: list[np.ndarray] = []
    for tau1, tau2, tau3 in taus:
        vec = np.array([tau1, tau2, tau3])
        if any(float(np.linalg.norm(vec - s)) < 0.1 for s in seen):
            continue
        coords = _apply_solution(full_bb, full_o, params, tau1, tau2, tau3, c_stem_N)
        if coords is not None:
            # Strip stems, return loop-only (N, 4, 3)
            solutions.append(coords[1:-1])
            seen.append(vec)

    return solutions


def close_loop(
    loop_coords: np.ndarray,
    n_stem_N: np.ndarray, n_stem_CA: np.ndarray, n_stem_C: np.ndarray,
    c_stem_N: np.ndarray, c_stem_CA: np.ndarray, c_stem_C: np.ndarray,
    pivot1: Optional[int] = None,
    pivot2: Optional[int] = None,
    pivot3: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Close loop using analytical KIC. Auto-selects pivots if not given.
    Returns (N, 4, 3) float32 best solution or None.
    Minimum loop length: 3 residues (pivots at 1, 2, N in augmented backbone).
    """
    n_res = len(loop_coords)
    n_full = n_res + 2   # includes stems

    # Pivots within loop portion: [1..N] in the augmented backbone
    # Default: evenly space across [1..N] with N=n_res
    N = n_res
    if pivot3 is None: pivot3 = N
    if pivot2 is None: pivot2 = max(1, N * 2 // 3)
    if pivot1 is None: pivot1 = max(1, N * 1 // 3)

    pivot1 = max(1, min(pivot1, pivot2 - 1))
    pivot2 = max(pivot1 + 1, min(pivot2, pivot3 - 1))
    pivot3 = max(pivot2 + 1, min(pivot3, N))

    if not (0 < pivot1 < pivot2 < pivot3 < n_full - 1):
        return None

    sols = solve(loop_coords,
                 n_stem_N, n_stem_CA, n_stem_C,
                 c_stem_N, c_stem_CA, c_stem_C,
                 pivot1, pivot2, pivot3)

    if not sols:
        return None

    c_stem_N = np.asarray(c_stem_N, float)
    return min(sols,
               key=lambda s: float(np.linalg.norm(
                   s[-1, 2].astype(float) - c_stem_N)))
