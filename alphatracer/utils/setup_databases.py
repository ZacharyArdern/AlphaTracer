"""Download ProMod3 loop databases on first use.

Run standalone:
    python setup_databases.py [--data-dir /path/to/data]

Or import and call ensure_promod3_databases() before loading ProMod3.
"""
import os
import sys
import time
import hashlib
import argparse
import urllib.request
from pathlib import Path

# ProMod3 3.6.0 database files on GitLab
_BASE_URL = (
    "https://git.scicore.unibas.ch/schwede/ProMod3/-/raw/3.6.0/loop/data"
)

# (remote filename, local filename, expected size bytes)
LOOP_DB_FILES = [
    ("portable_frag_db.dat",                "frag_db.dat",                302_245_897),
    ("portable_structure_db.dat",           "structure_db.dat",            521_252_864),
    ("portable_torsion_sampler_coil.dat",   "torsion_sampler_coil.dat",      8_781_048),
    ("portable_torsion_sampler.dat",        "torsion_sampler.dat",           8_781_048),
    ("portable_torsion_sampler_helical.dat","torsion_sampler_helical.dat",   8_781_048),
    ("portable_torsion_sampler_extended.dat","torsion_sampler_extended.dat", 8_781_048),
]

SCORING_DB_FILES = [
    # smaller files — bundled in the ProMod3 image, listed for completeness
]


def default_data_dir() -> Path:
    """Return the ProMod3 shared data directory.

    Priority:
    1. PROMOD3_SHARED_DATA_PATH env var
    2. /usr/local/share/promod3  (Docker / installed path)
    3. ~/.alphatracer/promod3    (user-writable fallback)
    """
    env = os.environ.get("PROMOD3_SHARED_DATA_PATH")
    if env:
        return Path(env)
    system = Path("/usr/local/share/promod3")
    if system.exists() and os.access(system, os.W_OK):
        return system
    user = Path.home() / ".alphatracer" / "promod3"
    return user


def _progress_hook(label: str):
    start = time.perf_counter()
    last = [0]

    def hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb  = downloaded / 1e6
            tot = total_size / 1e6
            elapsed = time.perf_counter() - start
            speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
            if pct != last[0]:
                print(f"\r  {label}: {pct:3d}%  {mb:.0f}/{tot:.0f} MB  "
                      f"{speed:.1f} MB/s", end="", flush=True)
                last[0] = pct
    return hook


def ensure_promod3_databases(data_dir: Path = None,
                              loop_only: bool = True,
                              verbose: bool = True) -> Path:
    """Download missing ProMod3 databases and return the data directory.

    Parameters
    ----------
    data_dir  : target directory; defaults to default_data_dir()
    loop_only : if True, only download loop DB files (frag_db, structure_db,
                torsion samplers) — scoring/sidechain data is bundled in image
    verbose   : print progress

    Returns
    -------
    Path to the data directory (for setting PROMOD3_SHARED_DATA_PATH)
    """
    if data_dir is None:
        data_dir = default_data_dir()

    loop_dir = Path(data_dir) / "loop_data"
    loop_dir.mkdir(parents=True, exist_ok=True)

    files = LOOP_DB_FILES if loop_only else LOOP_DB_FILES + SCORING_DB_FILES

    for remote_name, local_name, expected_size in files:
        dest = loop_dir / local_name
        if dest.exists():
            actual = dest.stat().st_size
            if actual == expected_size:
                if verbose:
                    print(f"  {local_name}: already present ({actual/1e6:.0f} MB)")
                continue
            else:
                if verbose:
                    print(f"  {local_name}: size mismatch "
                          f"({actual} vs {expected_size}), re-downloading")
                dest.unlink()

        url = f"{_BASE_URL}/{remote_name}"
        if verbose:
            print(f"  Downloading {local_name} ({expected_size/1e6:.0f} MB)...")
        try:
            urllib.request.urlretrieve(
                url, dest,
                reporthook=_progress_hook(local_name) if verbose else None)
            if verbose:
                print()  # newline after progress bar
        except Exception as e:
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Failed to download {remote_name}: {e}") from e

        # Verify size
        actual = dest.stat().st_size
        if actual != expected_size:
            dest.unlink()
            raise RuntimeError(
                f"{local_name}: downloaded {actual} bytes, expected {expected_size}")

    if verbose:
        total = sum(f.stat().st_size for f in loop_dir.iterdir()) / 1e9
        print(f"\nProMod3 databases ready at {loop_dir}  ({total:.2f} GB total)")

    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ProMod3 databases")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Target directory (default: auto-detect)")
    args = parser.parse_args()
    data_dir = ensure_promod3_databases(data_dir=args.data_dir, verbose=True)
    print(f"\nSet PROMOD3_SHARED_DATA_PATH={data_dir} before running ProMod3.")
