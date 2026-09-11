"""
Native fragment database query — no ProMod3/OST required.

Loads the fragdb.npz produced by scripts/setup/dump_fragdb.py and provides
fast stem-geometry-based lookup of backbone fragment candidates.

Public API
----------
load(path)
    Load the database from a .npz file. Returns a FragDB instance.

FragDB.query(n_stem_N, n_stem_CA, n_stem_C,
             c_stem_N, c_stem_CA, c_stem_C,
             frag_len, extra_bins=1)
    Return (coords, seqs) for matching fragments.
    coords : float32 (n_candidates, frag_len, 4, 3)  — N/CA/C/O per residue
    seqs   : list of str, length n_candidates

Coordinate convention
---------------------
All inputs are numpy arrays of shape (3,) in Ångströms.
n_stem_* : N-terminal anchor residue (residue BEFORE the loop)
c_stem_* : C-terminal anchor residue (residue AFTER the loop)

Returned coords are in absolute Ångström space (float32), reconstructed by
rotating from the stored n_stem local frame back to the query frame.

Stem geometry
-------------
Reimplements ProMod3's StemPairOrientation (core/src/geom_stems.cc):
  distance  : ||c_stem_N − n_stem_C||
  angle_one : signed angle in n_stem plane
  angle_two : signed angle from plane to N_C vector
  angle_three, angle_four : mirror quantities relative to c_stem

Storage formats (auto-detected)
---------------------------------
float16  (fragdb_local.npz):
  coords_{fl}  float16 (N_frags_fl, fl, 4, 3)  — n_stem local frame, absolute

int16+delta  (fragdb_int16delta.npz):
  coords_{fl}  int16   (N_frags_fl, fl, 4, 3)  — n_stem local frame, residue-delta encoded
  coord_scale  float32 scalar                  — multiply int16 → Å before cumsum
  Decoding: coords_Å = cumsum(int16 × coord_scale, axis=1)  (cumsum along residue axis)
  This stores inter-residue differences, which are small and compress well.

Both formats use lazy per-fl loading: only keys+offsets (~5% of data) are
loaded at startup; coord and seq arrays are decompressed on first query per
fragment-length, then cached. Most runs use 3-5 of 12 fragment lengths, so
effective startup is ~0.1s rather than ~15s.

CSR-style per fragment-length (fl):
  keys_{fl}    int16  (N_keys_fl, 5)           geometry bins [d,a1,a2,a3,a4]
  offsets_{fl} int32  (N_keys_fl+1,)           row offsets into flat arrays
  coords_{fl}  float16|int16 (N_frags_fl, fl, 4, 3)
  seqs_{fl}    object  (N_frags_fl,)
"""

from __future__ import annotations

import threading
import numpy as np
from pathlib import Path
from typing import Optional

# ── Geometry ──────────────────────────────────────────────────────────────────

def _signed_angle(v1: np.ndarray, v2: np.ndarray, ref: np.ndarray) -> float:
    """Signed angle between v1 and v2, with sign given by ref axis."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    v1 = v1 / n1
    v2 = v2 / n2
    cross = np.cross(v1, v2)
    cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    sin_a = float(np.linalg.norm(cross))
    angle = np.arctan2(sin_a, cos_a)
    if np.dot(cross, ref) < 0:
        angle = -angle
    return float(angle)


def stem_pair_orientation(
    n_N: np.ndarray, n_CA: np.ndarray, n_C: np.ndarray,
    c_N: np.ndarray, c_CA: np.ndarray, c_C: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Compute the 5 geometric features describing a stem pair.

    Mirrors ProMod3 StemPairOrientation::Init() exactly.

    Returns (distance, angle_one, angle_two, angle_three, angle_four)
    where angles are in radians.
    """
    NC_vec = c_N - n_C
    distance = float(np.linalg.norm(NC_vec))

    def _normal(a, b, c):
        v = np.cross(a - b, c - b)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    norm1 = _normal(n_N, n_CA, n_C)
    norm2 = _normal(c_N, c_CA, c_C)

    c_N_onto_p1 = c_N - np.dot(NC_vec, norm1) * norm1
    v1_to_cN    = c_N_onto_p1 - n_C

    ref = norm1
    angle_one = _signed_angle(v1_to_cN, n_C - n_CA, ref)

    ref = np.cross(norm1, v1_to_cN)
    angle_two = _signed_angle(v1_to_cN, NC_vec, ref)

    neg_NC = -NC_vec
    n_C_onto_p2 = n_C - np.dot(neg_NC, norm2) * norm2
    v2_to_nC    = n_C_onto_p2 - c_N

    ref = norm2
    angle_three = _signed_angle(v2_to_nC, c_N - c_CA, ref)

    ref = np.cross(norm2, v2_to_nC)
    angle_four = _signed_angle(v2_to_nC, neg_NC, ref)

    return distance, angle_one, angle_two, angle_three, angle_four


def _local_frame(n_N: np.ndarray, n_CA: np.ndarray, n_C: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix (rows = basis vectors) matching dump convention."""
    x = n_C - n_CA
    x = x / np.linalg.norm(x)
    z = np.cross(x, n_N - n_CA)
    nz = np.linalg.norm(z)
    z = z / nz if nz > 1e-9 else np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z]).astype(np.float32)


def _bin_key_5(
    n_N: np.ndarray, n_CA: np.ndarray, n_C: np.ndarray,
    c_N: np.ndarray, c_CA: np.ndarray, c_C: np.ndarray,
    dist_bin: float,
    ang_bin: float,
) -> tuple[int, int, int, int, int]:
    """Discretise stem geometry into 5-integer bin key (no frag_len)."""
    n_ang_bins = int(round(360 / ang_bin))
    dist, a1, a2, a3, a4 = stem_pair_orientation(n_N, n_CA, n_C, c_N, c_CA, c_C)
    d_bin  = int(round(dist / dist_bin))
    a1_bin = int(round(np.degrees(a1) / ang_bin)) % n_ang_bins
    a2_bin = int(round(np.degrees(a2) / ang_bin)) % n_ang_bins
    a3_bin = int(round(np.degrees(a3) / ang_bin)) % n_ang_bins
    a4_bin = int(round(np.degrees(a4) / ang_bin)) % n_ang_bins
    return (d_bin, a1_bin, a2_bin, a3_bin, a4_bin)


def _encode_key(d: int, a1: int, a2: int, a3: int, a4: int) -> int:
    """Pack 5 bin integers into a single int64 for fast searchsorted lookup.
    d uses bits 0-7 (max 255 = 25.5 Å); each angle uses 5 bits (max 31 >= 18 bins)."""
    return d | (a1 << 8) | (a2 << 13) | (a3 << 18) | (a4 << 23)


def _encode_keys_arr(keys_arr: np.ndarray) -> np.ndarray:
    """Vectorised encode of (N,5) int16 keys array → int64 array."""
    k = keys_arr.astype(np.int64)
    return k[:, 0] | (k[:, 1] << 8) | (k[:, 2] << 13) | (k[:, 3] << 18) | (k[:, 4] << 23)


# ── FragDB class ──────────────────────────────────────────────────────────────

class FragDB:
    """
    Fragment database with lazy per-fl loading and binary-search lookup.

    Lookup uses sorted int64-encoded keys + np.searchsorted instead of a
    Python dict, reducing startup from ~4s to ~0.05s. Only keys+offsets are
    loaded at construction; coord and seq arrays are decompressed on first
    query per fragment-length, then cached.

    Supports float16 (absolute) and int16+delta coord formats (auto-detected).
    """

    def __init__(
        self,
        # Per-fl: sorted encoded int64 keys + corresponding offsets array
        sorted_keys:  dict[int, np.ndarray],   # fl → int64 (N_keys,) sorted
        offsets:      dict[int, np.ndarray],   # fl → int32 (N_keys+1,)
        npz:          object,                  # open NpzFile for lazy loads
        dist_bin:     float,
        ang_bin:      float,
        coord_scale:  Optional[float],
        n_frags:      dict[int, int],
    ):
        self._sorted_keys = sorted_keys
        self._offsets     = offsets
        self._npz         = npz
        self._dist_bin    = dist_bin
        self._ang_bin     = ang_bin
        self._n_ang_bins  = int(round(360 / ang_bin))
        self._coord_scale = coord_scale
        self._n_frags     = n_frags

        self._coords: dict[int, np.ndarray] = {}
        self._seqs:   dict[int, np.ndarray] = {}
        self._locks: dict[int, threading.Lock] = {
            fl: threading.Lock() for fl in sorted_keys
        }

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _load_fl(self, fl: int) -> None:
        """Decompress and cache coords+seqs for one fragment length."""
        with self._locks[fl]:
            if fl in self._coords:
                return
            raw   = self._npz[f"coords_{fl}"]
            seqs  = self._npz[f"seqs_{fl}"]
            if self._coord_scale is not None:
                # int16 × scale → float32, then undo inter-residue delta encoding
                arr = raw.astype(np.float32) * self._coord_scale
                arr = np.cumsum(arr, axis=1)  # axis=1 is the residue axis
            else:
                arr = raw.astype(np.float32)
            self._coords[fl] = arr
            self._seqs[fl]   = seqs

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        n_stem_N:  np.ndarray,
        n_stem_CA: np.ndarray,
        n_stem_C:  np.ndarray,
        c_stem_N:  np.ndarray,
        c_stem_CA: np.ndarray,
        c_stem_C:  np.ndarray,
        frag_len: int,
        extra_bins: int = 1,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find fragments matching the given stem geometry.

        Parameters
        ----------
        n_stem_N/CA/C : (3,) float arrays — N-terminal anchor atom positions
        c_stem_N/CA/C : (3,) float arrays — C-terminal anchor atom positions
        frag_len      : number of residues to fill
        extra_bins    : neighbour radius in bin space (0=exact, 1=±1 each dim)

        Returns
        -------
        coords : float32 (n_candidates, frag_len, 4, 3) — N/CA/C/O per residue,
                 in absolute Ångström space
        seqs   : list[str] of length n_candidates
        """
        if frag_len not in self._sorted_keys:
            return np.empty((0, frag_len, 4, 3), dtype=np.float32), []

        center = _bin_key_5(
            n_stem_N, n_stem_CA, n_stem_C,
            c_stem_N, c_stem_CA, c_stem_C,
            self._dist_bin, self._ang_bin,
        )

        sorted_keys_fl = self._sorted_keys[frag_len]
        offsets_fl     = self._offsets[frag_len]   # (N_keys, 2) int32: [start, end]
        keys_to_check  = self._neighbour_keys(center, extra_bins)

        # Encode all neighbour keys at once, look up via searchsorted
        encoded = np.array(
            [_encode_key(*k) for k in keys_to_check], dtype=np.int64
        )
        idx = np.searchsorted(sorted_keys_fl, encoded)
        valid_mask = (idx < len(sorted_keys_fl)) & (sorted_keys_fl[idx] == encoded)

        slices = []
        seen_idx: set[int] = set()
        for i in idx[valid_mask]:
            i = int(i)
            if i in seen_idx:
                continue
            seen_idx.add(i)
            slices.append((int(offsets_fl[i, 0]), int(offsets_fl[i, 1])))

        if not slices:
            return np.empty((0, frag_len, 4, 3), dtype=np.float32), []

        self._load_fl(frag_len)
        coords_fl = self._coords[frag_len]
        seqs_fl   = self._seqs[frag_len]

        rel = np.concatenate([coords_fl[s:e] for s, e in slices], axis=0)
        R = _local_frame(n_stem_N, n_stem_CA, n_stem_C)
        # stored: local = R_src @ (world - n_stem_C_src)
        # recover: world = R_query.T @ local + n_stem_C_query  (== local @ R + n_stem_C)
        all_coords = rel @ R + n_stem_C.astype(np.float32)
        all_seqs = [str(seqs_fl[i]) for s, e in slices for i in range(s, e)]
        return all_coords, all_seqs

    def _neighbour_keys(self, center: tuple, radius: int) -> list[tuple]:
        if radius == 0:
            return [center]
        d, a1, a2, a3, a4 = center
        nb = self._n_ang_bins
        keys = []
        for dd in range(-radius, radius + 1):
            for da1 in range(-radius, radius + 1):
                for da2 in range(-radius, radius + 1):
                    for da3 in range(-radius, radius + 1):
                        for da4 in range(-radius, radius + 1):
                            keys.append((
                                d + dd,
                                (a1 + da1) % nb,
                                (a2 + da2) % nb,
                                (a3 + da3) % nb,
                                (a4 + da4) % nb,
                            ))
        return keys

    # ── Stats ────────────────────────────────────────────────────────────────

    def __repr__(self):
        total = sum(self._n_frags.values())
        fmt = f"int16-delta×{self._coord_scale:.4f}" if self._coord_scale else "float16"
        return (f"FragDB(frag_lengths={sorted(self._sorted_keys)}, "
                f"total_frags={total:,}, fmt={fmt}, "
                f"dist_bin={self._dist_bin}Å, ang_bin={self._ang_bin}°)")


# ── Loader ────────────────────────────────────────────────────────────────────

_CACHE: dict[str, FragDB] = {}


def load(path: Optional[str] = None) -> FragDB:
    """
    Load a FragDB from a .npz file. Supports float16 and int16+delta formats.

    Startup loads only keys+offsets (~0.1s). Coord/seq arrays are decompressed
    lazily on first query per fragment-length.

    Prefers fragdb_int16delta.npz > fragdb_local.npz > fragdb.npz when path
    is not specified. Caches so repeated calls with the same path are free.
    """
    if path is None:
        data_dir = Path(__file__).parent.parent / "data"
        for name in ("fragdb_int16delta.npz", "fragdb_local.npz", "fragdb.npz"):
            p = data_dir / name
            if p.exists():
                path = str(p)
                break
        if path is None:
            raise FileNotFoundError(
                f"No fragment DB found in {data_dir}. "
                "Run scripts/setup/dump_fragdb.py to generate it."
            )

    path = str(path)
    if path in _CACHE:
        return _CACHE[path]

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Fragment DB not found at {path}.\n"
            f"Run scripts/setup/dump_fragdb.py inside the ProMod3 Docker "
            f"container to generate it."
        )

    # Keep NpzFile open — arrays decompressed lazily per query
    npz = np.load(path, allow_pickle=True)

    dist_bin     = float(npz["dist_bin_size"])
    ang_bin      = float(npz["angular_bin_size"])
    frag_lengths = list(npz["frag_lengths"])
    coord_scale  = float(npz["coord_scale"]) if "coord_scale" in npz else None

    sorted_keys: dict[int, np.ndarray] = {}
    offsets:     dict[int, np.ndarray] = {}
    n_frags:     dict[int, int]        = {}

    # Load only keys + offsets eagerly — encode to int64 and sort for searchsorted.
    # ~2.9M keys × 8 bytes = ~23 MB total; builds in <0.1s vs ~4s for Python dicts.
    for fl in frag_lengths:
        if f"keys_{fl}" not in npz:
            continue
        keys_arr    = npz[f"keys_{fl}"]    # (N_keys, 5) int16
        offsets_arr = npz[f"offsets_{fl}"] # (N_keys+1,) int32

        encoded = _encode_keys_arr(keys_arr)  # (N_keys,) int64
        order   = np.argsort(encoded, kind='stable')
        sorted_keys[fl] = encoded[order]
        # Re-order offsets to match sorted key order; keep as start/end pairs
        # by building a new offsets array from sorted fragment counts
        counts = offsets_arr[1:] - offsets_arr[:-1]  # fragments per key
        sorted_counts = counts[order]
        new_offsets = np.zeros(len(sorted_counts) + 1, dtype=np.int32)
        np.cumsum(sorted_counts, out=new_offsets[1:])
        # Map sorted key i → original offset range, then remap to new positions.
        # Simpler: store original offsets reordered alongside sorted keys.
        # We need (start, end) for each sorted key; store as (N_keys+1,) where
        # sorted_offsets[i]:sorted_offsets[i+1] gives original start:end.
        orig_starts = offsets_arr[:-1][order]
        orig_ends   = offsets_arr[1:][order]
        combined = np.empty(len(order) + 1, dtype=np.int64)
        combined[:-1] = orig_starts
        combined[-1]  = orig_ends[-1]   # sentinel (not directly used)
        # Store as two parallel arrays for fast slicing
        sorted_keys[fl] = encoded[order]
        offsets[fl]     = np.stack([orig_starts, orig_ends], axis=1)  # (N_keys, 2) int32
        n_frags[fl]     = int(offsets_arr[-1])

    db = FragDB(sorted_keys, offsets, npz, dist_bin, ang_bin, coord_scale, n_frags)
    _CACHE[path] = db
    return db
