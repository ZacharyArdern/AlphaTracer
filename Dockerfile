# AlphaTracer 0.2 — Docker image
#
# Layers on the ProMod3 image (OST 2.11.1 + ProMod3 3.6.0 + all loop databases).
# Adds AlphaTracer dependencies and code.
#
# MLX is NOT included — Metal/MPS requires Apple Silicon host.
# On GPU hosts (Colab/cloud), structure prediction uses PyTorch + CUDA.
# On CPU-only hosts, PyTorch CPU is used (slower).
#
# Build:
#   docker build -t alphatracer:0.2 .
#
# Run (weights mounted from host):
#   docker run --rm \
#     -v /path/to/weights:/weights \
#     -v $(pwd):/work -w /work \
#     alphatracer:0.2 \
#     python /alphatracer/run_alphatracer.py -i proteins.fasta --backend cuda
#
# Run on CPU only:
#   docker run --rm -v $(pwd):/work -w /work alphatracer:0.2 \
#     python /alphatracer/run_alphatracer.py -i proteins.fasta --backend cpu
#
# Colab (Singularity):
#   singularity pull alphatracer.sif docker://ghcr.io/<user>/alphatracer:0.2
#   singularity exec --nv alphatracer.sif \
#     python /alphatracer/run_alphatracer.py -i proteins.fasta --backend cuda

FROM registry.scicore.unibas.ch/schwede/promod3:3.6.0-OST2.11.1-jammy

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git \
        diamond \
        libopenmm-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Install into the same Python that OST/ProMod3 uses (system python3)
RUN pip3 install --no-cache-dir \
    polars \
    parasail-python \
    pycurl \
    gemmi \
    biopython \
    numpy \
    scipy \
    "openmm>=8.0" \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ESM2 for OFS scoring (CPU/CUDA, no Metal needed)
RUN pip3 install --no-cache-dir fair-esm

# ── AlphaTracer code ───────────────────────────────────────────────────────────
COPY . /alphatracer/
WORKDIR /alphatracer

# ── Weight volume mount point ──────────────────────────────────────────────────
# Mount weights here: docker run -v /local/weights:/weights ...
# Expected layout:
#   /weights/
#     minifold_cache/
#       minifold_48L.ckpt              (PyTorch weights)
#       checkpoints/
#         mlx-esm2_t36_3B_UR50D_finetuned/   (MLX ESM2 — optional, Mac only)
#     minifold_mlx_48L/               (MLX MiniFormer — optional, Mac only)
RUN mkdir -p /weights

# ── ProMod3 database location ──────────────────────────────────────────────────
# Databases are already baked into the base image at:
#   /usr/local/share/promod3/loop_data/
# PROMOD3_SHARED_DATA_PATH is not set — ProMod3 finds them automatically.
# To use a mounted volume instead (saves ~860 MB in image size):
#   docker run -v /local/promod3_data:/usr/local/share/promod3 ...

# ── Environment ───────────────────────────────────────────────────────────────
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONPATH=/alphatracer

# ── Default entrypoint ────────────────────────────────────────────────────────
ENTRYPOINT ["python3", "/alphatracer/run_alphatracer.py"]
CMD ["--help"]
