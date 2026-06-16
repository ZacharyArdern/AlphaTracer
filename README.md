# AlphaTracer

AlphaTracer is a homology-guided protein structure prediction pipeline. It classifies query sequences into four tiers based on their relationship to structures in the AlphaFold Database (AFDB) and ESM Atlas, applying the most accurate method available for each tier.

## How it works

Each query sequence is processed through four classes in order. Once a sequence is handled by a class, it does not proceed to subsequent classes.

### Class A — Direct backbone graft
The query has a high-confidence AFDB or ESM Atlas hit: no insertions or deletions within the aligned region, and every 40-residue window achieves ≥80% sequence similarity. Backbone atoms are copied directly from the reference structure. Fast and highly accurate.

### Class B — Graft with loop closure
The query has an AFDB or ESM Atlas hit with 1–3 small indels (up to 5 residues each) and >30% overall identity. Matched positions are grafted from the reference; inserted residues are placed in an extended conformation; gaps are closed with Cyclic Coordinate Descent (CCD); and the result is refined by OpenMM minimisation.

### Class C — Domain-level template matching
No full-length hit qualifies, but the reference structure can be decomposed into rigid domains (via PAE graph clustering). Domains with ≥50% sequence identity to the query are grafted individually. Remaining query regions can optionally be filled in using MiniFold predictions, placed by L-BFGS-B, and closed with CCD.

### Class D — De novo prediction
No suitable homology exists. The full sequence is predicted by MiniFold (MLX backend on Apple Silicon; PyTorch on Linux/cloud), with up to two recycling rounds if mean pLDDT < 85. Sequences longer than 800 residues are skipped by default due to memory limits.

---

## Installation

**Python ≥ 3.11 required.**

```bash
git clone <repo>
cd AlphaTracer

# Apple Silicon (MLX backend)
pip install -e ".[mlx]"

# Linux / cloud (PyTorch + CUDA)
pip install -e ".[cuda]"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback
pip install -e ".[cpu]"
```

**Rust toolchain** is required for the kmer sketch search (Classes A–C). Binaries are compiled automatically on first run (~30 s) and cached to `~/.cache/alphatracer/`.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Structure database** — AlphaTracer searches against AFDB and/or ESM Atlas. Place the sequence parquet (e.g. `afdb_v6_reps.pq`, or a merged AFDB+ESMAtlas parquet) in `~/Science/Data/AFDB/` or point to it with the `AT_AFDB_DIR` environment variable. The kmer sketch index (`.sidx`) is built automatically from the parquet on first use. If the parquet contains a `db_type` column, AlphaTracer auto-detects the merged database and queries both AFDB and ESM Atlas for structures.

---

## Usage

```bash
# Basic run
alphatracer -i proteins.fasta

# Specify database directory
alphatracer -i proteins.fasta --dbdir /path/to/db

# Use a custom or merged parquet (e.g. AFDB + ESM Atlas)
alphatracer -i proteins.fasta --sketch-db /path/to/merged.pq

# Use DIAMOND BLASTP instead of kmer search for Class A
alphatracer -i proteins.fasta --diamond

# Skip homology classes, run de novo only
alphatracer -i proteins.fasta --skip-classA --skip-classB --skip-classC

# Explicit backend
alphatracer -i proteins.fasta --backend cuda    # Linux/cloud
alphatracer -i proteins.fasta --backend mlx     # Apple Silicon

# More threads, custom output directory
alphatracer -i proteins.fasta -t 8 --outdir my_results/
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | — | Input FASTA (required) |
| `--dbdir` | cwd | Directory containing the sequence parquet and sketch index |
| `--sketch-db` | — | Path to a custom parquet (overrides `--dbdir`; merged AFDB+ESMAtlas auto-detected) |
| `-t` | 4 | CPU threads |
| `--backend` | auto | `mlx`, `cuda`, or `cpu` |
| `--max-seq-len` | 800 | Skip Class D for sequences longer than this |
| `--no-fill-missing` | — | Disable MiniFold infill for Class C gaps |
| `--no-classD` | — | Skip de novo prediction entirely |
| `--verbose` | — | Full output instead of progress bar |

---

## Dependencies

Core: `polars`, `pyarrow`, `duckdb`, `biopython`, `parasail`, `gemmi`, `pycurl`, `numpy`, `scipy`, `igraph`, `openmm`

Backend-specific: `mlx` + `minifold-mlx` (Apple Silicon) or `torch` + `fair-esm` (Linux/cloud)

Optional: [ProMod3](https://openstructure.org/promod3/) for fragment-database loop closing (`--loop-closer promod3`); [DIAMOND](https://github.com/bbuchfink/diamond) for BLASTP-based Class A search (`--diamond`).

---

## External Links

- [MiniFold](https://github.com/jwohlwend/minifold)
- [MiniFold-MLX (macOS)](https://github.com/ZacharyArdern/MiniFold-MLX)
- [AlphaFold Database (AFDB)](https://alphafold.ebi.ac.uk/)
- [Biohub ESM Atlas](https://biohub.ai/esm/protein/atlas)
