#!/usr/bin/env python3
"""
dump_fragdb.py — Export ProMod3 fragment database to portable numpy format.

Run once inside the ProMod3 Docker container:

    docker run --rm --entrypoint /bin/bash \\
      --platform linux/amd64 \\
      -v /path/to/AlphaTracer_working/AT_dev:/work \\
      registry.scicore.unibas.ch/schwede/promod3:3.6.0-OST2.11.1-jammy \\
      -c "python3 /work/alphatracer/scripts/setup/dump_fragdb.py \\
          --out /work/alphatracer/data/fragdb.npz"

Memory notes
------------
The default --stride 4 processes every 4th chain (~5250 of 21119), yielding
~11M fragments and peak RAM ~7 GB during packing. Use --stride 1 for a
complete dump if ≥32 GB available in Docker.

Output format: fragdb.npz (CSR-style, per fragment-length)
-----------------------------------------------------------
Global scalars:
  dist_bin_size    float32   (Å per bin)
  angular_bin_size float32   (degrees per bin)
  frag_lengths     int32[N]  fragment lengths present
  stride           int32     chain stride used
  n_chains_total   int32
  n_chains_used    int32

Per fragment-length fl (fl = 3..14 by default):
  keys_{fl}    int16   (N_keys_fl, 5)          geometry bins: [d, a1, a2, a3, a4]
  offsets_{fl} int32   (N_keys_fl + 1,)        CSR row offsets into flat arrays
  coords_{fl}  float16 (N_frags_fl, fl, 4, 3)  N/CA/C/O per residue, in the
                                               n_stem local orientation frame
  seqs_{fl}    object  (N_frags_fl,)           one-letter sequences (one per frag)

Coordinate convention
---------------------
coords_{fl} stores coords in the local frame of each fragment's n_stem:
  origin = n_stem_C
  x-axis = CA→C bond direction
  z-axis = normal to N-CA-C plane
  y-axis = z × x

At query time: rotate query n_stem to the same canonical frame, then
inverse-rotate returned coords back to the query's absolute frame.
float16 precision is ~0.03 Å at typical intra-loop distances.

Query (see alphatracer/utils/frag_db.py):
  For key (d,a1,a2,a3,a4), find its index i in keys_{fl}, then
  coords = coords_{fl}[offsets_{fl}[i] : offsets_{fl}[i+1]]
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

try:
    from promod3 import loop as pm3_loop
    from promod3.core import StemCoords, StemPairOrientation
except ImportError as e:
    print(f"ERROR: ProMod3/OST not importable. Run inside the Docker container.\n{e}")
    sys.exit(1)

# Bin sizes must match ProMod3 defaults (frag_db.cc)
DIST_BIN_SIZE = 1.0   # Å
ANG_BIN_SIZE  = 20    # degrees
N_ANG_BINS    = 360 // ANG_BIN_SIZE  # 18

MAX_FRAGS_PER_KEY = 200


def vec3(v):
    return np.array([v[0], v[1], v[2]], dtype=np.float32)


def stem_key(bb, n_idx, c_idx, frag_len):
    n_sc = StemCoords()
    n_sc.n_coord  = bb.GetN(n_idx)
    n_sc.ca_coord = bb.GetCA(n_idx)
    n_sc.c_coord  = bb.GetC(n_idx)

    c_sc = StemCoords()
    c_sc.n_coord  = bb.GetN(c_idx)
    c_sc.ca_coord = bb.GetCA(c_idx)
    c_sc.c_coord  = bb.GetC(c_idx)

    o = StemPairOrientation(n_sc, c_sc)

    d  = int(round(o.distance / DIST_BIN_SIZE))
    a1 = int(round(np.degrees(o.angle_one)   / ANG_BIN_SIZE)) % N_ANG_BINS
    a2 = int(round(np.degrees(o.angle_two)   / ANG_BIN_SIZE)) % N_ANG_BINS
    a3 = int(round(np.degrees(o.angle_three) / ANG_BIN_SIZE)) % N_ANG_BINS
    a4 = int(round(np.degrees(o.angle_four)  / ANG_BIN_SIZE)) % N_ANG_BINS

    return (d, a1, a2, a3, a4, frag_len)


def _local_frame(n_N: np.ndarray, n_CA: np.ndarray, n_C: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix (rows = basis vectors) for n_stem local frame.
    x: CA→C bond; z: normal to N-CA-C plane; y: completes right-hand frame."""
    x = n_C - n_CA
    x /= np.linalg.norm(x)
    z = np.cross(x, n_N - n_CA)
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        z /= nz
    y = np.cross(z, x)
    return np.stack([x, y, z]).astype(np.float32)  # (3,3): local = R @ (world - origin)


def extract_coords_local(bb, start, length,
                         n_N: np.ndarray, n_CA: np.ndarray, n_C: np.ndarray) -> np.ndarray:
    """Return float16 (length, 4, 3) coords in n_stem local orientation frame."""
    out = np.empty((length, 4, 3), dtype=np.float32)
    for i in range(length):
        j = start + i
        out[i, 0] = vec3(bb.GetN(j))
        out[i, 1] = vec3(bb.GetCA(j))
        out[i, 2] = vec3(bb.GetC(j))
        out[i, 3] = vec3(bb.GetO(j))
    R = _local_frame(n_N, n_CA, n_C)
    # Translate to n_stem_C origin, rotate into local frame
    return ((out - n_C) @ R.T).astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/work/alphatracer/data/fragdb.npz")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4,
                    help="Process every Nth chain (default 4; use 1 for full dump)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frag_lengths = list(range(args.min_len, args.max_len + 1))

    print("Loading StructureDB...", flush=True)
    t0 = time.perf_counter()
    struct_db = pm3_loop.LoadStructureDB()
    n_chains = struct_db.GetNumCoords()
    print(f"  {n_chains:,} chains in {time.perf_counter()-t0:.1f}s", flush=True)

    try:
        frag_db = pm3_loop.LoadFragDB()
        frag_lengths = [l for l in frag_lengths if frag_db.HasFragLength(l)]
        print(f"  Fragment lengths confirmed from FragDB: {frag_lengths}", flush=True)
    except Exception:
        print(f"  Using lengths: {frag_lengths}", flush=True)

    chain_indices = list(range(0, n_chains, args.stride))
    print(f"  Stride={args.stride} → {len(chain_indices):,} of {n_chains:,} chains\n",
          flush=True)

    # ── Memory-efficient collection ────────────────────────────────────────────
    # coords_buf[key] = bytearray of packed float32 coords (avoids per-array overhead)
    # seqs_buf[key]   = list of sequence strings
    # counts_buf[key] = fragment count
    coords_buf: dict = {}
    seqs_buf:   dict = {}
    counts_buf: dict = {}

    bytes_per_frag = {fl: fl * 4 * 3 * 2 for fl in frag_lengths}  # float16 = 2 bytes

    n_frags = 0
    n_skip  = 0

    print(f"Iterating {len(chain_indices):,} chains × {len(frag_lengths)} lengths...",
          flush=True)
    t0 = time.perf_counter()

    for batch_i, ci in enumerate(chain_indices):
        info = struct_db.GetCoordInfo(ci)
        chain_len = info.size
        if chain_len < min(frag_lengths) + 2:
            continue

        try:
            bb = struct_db.GetBackboneList(ci)
        except Exception:
            n_skip += 1
            continue

        seq = bb.GetSequence()

        for frag_len in frag_lengths:
            bpf = bytes_per_frag[frag_len]
            for j in range(1, chain_len - frag_len):
                try:
                    key = stem_key(bb, j - 1, j + frag_len, frag_len)
                except Exception:
                    n_skip += 1
                    continue

                if counts_buf.get(key, 0) >= MAX_FRAGS_PER_KEY:
                    continue

                n_N  = vec3(bb.GetN(j - 1))
                n_CA = vec3(bb.GetCA(j - 1))
                n_C  = vec3(bb.GetC(j - 1))
                coords = extract_coords_local(bb, j, frag_len, n_N, n_CA, n_C)
                frag_seq = seq[j: j + frag_len]

                if key not in coords_buf:
                    coords_buf[key] = bytearray()
                    seqs_buf[key]   = []
                    counts_buf[key] = 0

                coords_buf[key] += coords.tobytes()
                seqs_buf[key].append(frag_seq)
                counts_buf[key] += 1
                n_frags += 1

        if (batch_i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (batch_i + 1) / elapsed
            eta  = (len(chain_indices) - batch_i - 1) / rate
            print(f"  {batch_i+1:>6,}/{len(chain_indices):,}  "
                  f"keys={len(coords_buf):,}  frags={n_frags:,}  "
                  f"ETA={eta/60:.1f}min", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nFinished: {n_frags:,} fragments, {len(coords_buf):,} unique keys, "
          f"{n_skip:,} skipped, {elapsed:.1f}s", flush=True)

    # ── Pack into per-fl CSR arrays ────────────────────────────────────────────
    # Process one frag_len at a time to keep peak RAM low.
    print("Packing arrays (per fragment length)...", flush=True)

    save_dict = {
        'dist_bin_size':    np.float32(DIST_BIN_SIZE),
        'angular_bin_size': np.float32(ANG_BIN_SIZE),
        'frag_lengths':     np.array(frag_lengths, dtype=np.int32),
        'stride':           np.int32(args.stride),
        'n_chains_total':   np.int32(n_chains),
        'n_chains_used':    np.int32(len(chain_indices)),
    }

    for fl in frag_lengths:
        # Collect all keys for this frag_len
        fl_keys = sorted(k for k in coords_buf if k[5] == fl)
        if not fl_keys:
            continue

        n_keys_fl = len(fl_keys)
        total_fl  = sum(counts_buf[k] for k in fl_keys)

        keys_arr    = np.empty((n_keys_fl, 5), dtype=np.int16)
        offsets_arr = np.zeros(n_keys_fl + 1, dtype=np.int32)
        coords_arr  = np.empty((total_fl, fl, 4, 3), dtype=np.float16)
        seqs_arr    = np.empty(total_fl, dtype=object)

        offset = 0
        for i, key in enumerate(fl_keys):
            n = counts_buf[key]
            raw = np.frombuffer(coords_buf[key], dtype=np.float16).reshape(n, fl, 4, 3)
            coords_arr[offset: offset + n] = raw
            seqs_arr[offset: offset + n]   = seqs_buf[key]
            keys_arr[i]    = key[:5]
            offsets_arr[i] = offset
            offset += n
        offsets_arr[n_keys_fl] = offset

        save_dict[f'keys_{fl}']    = keys_arr
        save_dict[f'offsets_{fl}'] = offsets_arr
        save_dict[f'coords_{fl}']  = coords_arr
        save_dict[f'seqs_{fl}']    = seqs_arr

        gb = coords_arr.nbytes / 1e9
        print(f"  fl={fl:2d}: {n_keys_fl:>7,} keys, {total_fl:>8,} frags  "
              f"({gb:.2f} GB)", flush=True)

    # Free collection buffers before the savez allocation
    del coords_buf, seqs_buf, counts_buf

    print(f"Saving to {out_path}...", flush=True)
    t_save = time.perf_counter()
    np.savez_compressed(out_path, **save_dict)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {size_mb:.0f} MB in {time.perf_counter()-t_save:.1f}s", flush=True)
    print(f"Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
