#!/usr/bin/env python3
"""
run_alphatracer.py  —  AlphaTracer pipeline wrapper (v0.2)

Runs the three component scripts in sequence:
  AT_classA.py  →  AT_classB.py  →  AT_classC_and_D.py

All component-script parameters are exposed here.  Shared parameters
(threads, CCD settings, OpenMM iters) are passed to every script that
accepts them.  Class-specific parameters are forwarded only to the
relevant script.

Usage
-----
  python run_alphatracer.py -i proteins.fasta [options]

  # Use ProMod3 fragment-DB loop closing instead of CCD:
  python run_alphatracer.py -i proteins.fasta --loop-closer promod3

  # Use CUDA backend for structure prediction (Colab/cloud):
  python run_alphatracer.py -i proteins.fasta --backend cuda

  # Use a different Python interpreter (e.g. a venv that has MLX):
  python run_alphatracer.py -i proteins.fasta \\
      --classcd-python /path/to/miniscaffold/.venv/bin/python

  # Skip stages that already completed:
  python run_alphatracer.py -i proteins.fasta --skip-classA --skip-classB

  # Class D de-novo only (no Class A/B/C):
  python run_alphatracer.py -i proteins.fasta \\
      --skip-classA --skip-classB --skip-classC
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).parent          # alphatracer/
_PKG_PARENT = str(_HERE.parent)              # repo root — added to PYTHONPATH


def _script(name: str) -> str:
    """Return the absolute path to a component script in the same directory."""
    return str(_HERE / name)


def _run(cmd: list[str], label: str, extra_env: dict | None = None) -> None:
    """Run a subprocess command, printing a header and raising on failure."""
    print()
    print('=' * 60)
    print(f'  AlphaTracer  ►  {label}')
    print('  Command: ' + ' '.join(cmd))
    print('=' * 60)
    env = {**os.environ, **(extra_env or {})}
    # Ensure the repo root is on PYTHONPATH so subprocesses find the alphatracer package.
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = _PKG_PARENT + (os.pathsep + existing if existing else '')
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f'\n[FATAL] {label} exited with code {result.returncode}. Aborting.')


def _flag(name: str, value) -> list[str]:
    """Return [--name, str(value)] if value is truthy, else []."""
    return [name, str(value)] if value else []


def _bool_flag(name: str, enabled: bool) -> list[str]:
    """Return [name] if enabled, else []."""
    return [name] if enabled else []


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='run_alphatracer.py',
        description='AlphaTracer — full pipeline wrapper (Class A → B → C/D)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        '-i', '--input', required=True,
        help='Input FASTA of query protein sequences',
    )

    # ── Shared ────────────────────────────────────────────────────────────────
    shared = p.add_argument_group('shared options (forwarded to all relevant scripts)')
    shared.add_argument(
        '--diamond', action='store_true', default=False,
        help='Use DIAMOND blastp for Class A search instead of kmer sketch (default: kmer)',
    )
    shared.add_argument(
        '-d', '--database',
        default=os.path.expanduser('~/Science/Data/AFDB/afdb_v6_reps.dmnd'),
        help='DIAMOND database for Class A (only used with --diamond)',
    )
    shared.add_argument('--top-k', type=int, default=5,
                        help='Hits per query from kmer search (default: 5)')
    shared.add_argument('-t', '--threads', type=int, default=4,
                        help='CPU threads')
    shared.add_argument('--mm-iters', type=int, default=300,
                        help='OpenMM minimisation iterations (Class B/C/D)')
    shared.add_argument('--ccd-iters', type=int, default=200,
                        help='Max CCD iterations per gap (Class B/C)')
    shared.add_argument('--ccd-tol', type=float, default=0.15,
                        help='CCD convergence tolerance in Å (Class B/C)')
    shared.add_argument('--flank', type=int, default=3,
                        help='Mobile flanking residues for CCD (Class B/C)')
    shared.add_argument('--loop-closer', default='ccd',
                        choices=['ccd', 'promod3'],
                        help='Loop closing backend: ccd (default) or promod3 '
                             '(fragment DB + CCD, requires OST/ProMod3)')
    shared.add_argument('--promod3-data-dir', default=None,
                        help='ProMod3 database directory '
                             '(default: $PROMOD3_SHARED_DATA_PATH or '
                             '/usr/local/share/promod3 or ~/.alphatracer/promod3)')

    # ── Backend ───────────────────────────────────────────────────────────────
    backend = p.add_argument_group('compute backend (for structure prediction)')
    backend.add_argument('--backend', default='auto',
                         choices=['auto', 'mlx', 'cuda', 'cpu'],
                         help='Inference backend for MLX MiniFold. '
                              'auto: use MLX on Apple Silicon, CUDA if available, '
                              'else CPU. mlx: force Apple Metal (Mac only). '
                              'cuda: force PyTorch CUDA (Colab/cloud). '
                              'cpu: force CPU (slow, universal).')

    # ── Class A ───────────────────────────────────────────────────────────────
    grp_a = p.add_argument_group('Class A options')
    grp_a.add_argument('--window-size', type=int, default=40,
                       help='Sliding window size for identity check')
    grp_a.add_argument('--pctsim', type=float, default=80.0,
                       help='Min %% sequence similarity per window')

    # ── Class B ───────────────────────────────────────────────────────────────
    grp_b = p.add_argument_group('Class B options')
    grp_b.add_argument('--max-indels', type=int, default=3,
                       help='Max number of indels allowed')
    grp_b.add_argument('--max-indel-len', type=int, default=5,
                       help='Max length of each indel')
    grp_b.add_argument('--b-min-pctsim', type=float, default=30.0,
                       help='Min identity for Class B candidates')
    grp_b.add_argument('--b-limit', type=int, default=0,
                       help='Process only first N Class B structures (0=all)')

    # ── Class C/D ─────────────────────────────────────────────────────────────
    grp_cd = p.add_argument_group('Class C/D options')
    grp_cd.add_argument('--c-min-pctsim', type=float, default=50.0,
                        help='Min identity for Class C candidates')
    grp_cd.add_argument('--pae-cutoff', type=float, default=5.0,
                        help='PAE graph edge weight cutoff (Å)')
    grp_cd.add_argument('--pae-power', type=float, default=1.0,
                        help='PAE graph weighting power')
    grp_cd.add_argument('--pae-resolution', type=float, default=1.0,
                        help='igraph community resolution for PAE domains')
    grp_cd.add_argument('--pae-resolution-large', type=float, default=2.0,
                        help='Resolution for reference proteins >--large-domain-threshold aa')
    grp_cd.add_argument('--large-domain-threshold', type=int, default=500,
                        help='Reference length above which --pae-resolution-large is used')
    grp_cd.add_argument('--c-limit', type=int, default=0,
                        help='Limit Class C to first N sequences (0=all)')
    grp_cd.add_argument('--fill-missing', action='store_true', default=False,
                        help='Fill non-domain query regions with OFS+MLX MiniFold (Class C)')
    grp_cd.add_argument('--min-frag-len', type=int, default=5,
                        help='Minimum fragment length to predict (Class C fill-missing)')
    grp_cd.add_argument('--lbfgsb-iters', type=int, default=300,
                        help='L-BFGS-B iterations for fragment placement (Class C)')
    grp_cd.add_argument('--mini-model-size', type=str, default='12L',
                        choices=['12L', '48L'],
                        help='MLX MiniFold model size (Class C/D)')
    grp_cd.add_argument('--anchor-k', type=float, default=1000.0,
                        help='Cα anchor restraint force constant kJ/mol/nm² (Class C)')
    grp_cd.add_argument('--plddt-threshold', type=float, default=85.0,
                        help='pLDDT threshold: recycle until mean pLDDT >= this (default: 85)')
    grp_cd.add_argument('--max-recyclings', type=int, default=2,
                        help='Max recycling rounds per sequence (default: 2)')
    grp_cd.add_argument('--ppl-short-threshold', type=int, default=100,
                        help='Sequences <= this length skip recycling (default: 100)')
    grp_cd.add_argument('--no-classD', action='store_true', default=False,
                        help='Skip Class D predictions')
    grp_cd.add_argument('--classD-limit', type=int, default=0,
                        help='Limit Class D to first N sequences (0=all)')
    grp_cd.add_argument('--batch-tokens', type=int, default=262144,
                        help='Max total tokens per MLX batch in Class D')
    grp_cd.add_argument('--max-seq-len', type=int, default=800,
                        help='Skip MLX prediction for sequences longer than this (0=no limit)')
    grp_cd.add_argument('--mm-iters-C', type=int, default=50,
                        help='OpenMM minimisation iterations for Class C/D (overrides --mm-iters; '
                             'default: 50, lower than Class B since AF structures are already clean)')

    # ── Pipeline control ──────────────────────────────────────────────────────
    ctrl = p.add_argument_group('pipeline control')
    ctrl.add_argument('--skip-classA', action='store_true', default=False,
                      help='Skip Class A (requires a prior run\'s output directory)')
    ctrl.add_argument('--skip-classB', action='store_true', default=False,
                      help='Skip Class B')
    ctrl.add_argument('--skip-classC', action='store_true', default=False,
                      help='Skip Class C')
    ctrl.add_argument('--python', default=sys.executable,
                      help='Python interpreter for Class A/B scripts')
    ctrl.add_argument('--classcd-python', default=None,
                      help='Python interpreter for Class C/D script '
                           '(defaults to --python; use the miniscaffold venv '
                           'if MLX/OFS are not in the main env)')

    return p.parse_args()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    input_path   = Path(args.input)
    proc_dir     = f'AT_processing_{input_path.stem}'
    py_main      = args.python
    py_cd        = args.classcd_python or py_main

    print('=' * 60)
    print('AlphaTracer  —  Full Pipeline Wrapper')
    print('=' * 60)
    search_method = 'diamond' if args.diamond else 'kmer'
    # When using kmer, min_pctsim thresholds are lowered to 1 because approx_pident
    # = shared hash count (already pre-filtered by the kmer index).
    b_min_pctsim = args.b_min_pctsim if args.diamond else 1
    c_min_pctsim = args.c_min_pctsim if args.diamond else 1

    print(f'  Input:        {args.input}')
    print(f'  Processing:   {proc_dir}/')
    print(f'  Search:       {search_method}')
    print(f'  Python (A/B): {py_main}')
    print(f'  Python (C/D): {py_cd}')
    print(f'  Loop closer:  {args.loop_closer}')
    print(f'  Backend:      {args.backend}')
    skips = [c for c, s in [('A', args.skip_classA), ('B', args.skip_classB),
                              ('C', args.skip_classC)] if s]
    skips += (['D'] if args.no_classD else [])
    print(f'  Skip:         {", ".join(skips) or "none"}')
    print()

    # ── ProMod3 database setup ─────────────────────────────────────────────────
    if args.loop_closer == 'promod3':
        from setup_databases import ensure_promod3_databases, default_data_dir
        data_dir = ensure_promod3_databases(
            data_dir=args.promod3_data_dir, verbose=True)
        os.environ['PROMOD3_SHARED_DATA_PATH'] = str(data_dir)

    # ── Class A ───────────────────────────────────────────────────────────────
    if not args.skip_classA:
        if args.diamond:
            # DIAMOND: single blocking call (no classify/download split)
            cmd_a = [
                py_main, _script('AT_classA.py'),
                '-i', args.input,
                '-d', args.database,
                '-t', str(args.threads),
                '--window-size', str(args.window_size),
                '--pctsim',      str(args.pctsim),
            ]
            _run(cmd_a, f'Class A ({search_method})')
            # Then run Class B normally below
            if not args.skip_classB:
                cmd_b = [
                    py_main, _script('AT_classB.py'),
                    '-i', proc_dir,
                    '-t', str(args.threads),
                    '--max-indels',    str(args.max_indels),
                    '--max-indel-len', str(args.max_indel_len),
                    '--min-pctsim',    str(b_min_pctsim),
                    '--mm-iters',      str(args.mm_iters),
                    '--ccd-iters',     str(args.ccd_iters),
                    '--ccd-tol',       str(args.ccd_tol),
                    '--flank',         str(args.flank),
                    '--loop-closer',   args.loop_closer,
                    *_flag('--promod3-data-dir', args.promod3_data_dir),
                    *_flag('--limit', args.b_limit),
                ]
                _run(cmd_b, 'Class B')
            else:
                print('[SKIP] Class B')
        else:
            # kmer: Phase 1 — classify only (~10s), writes classA.pq
            cmd_a_classify = [
                py_main, _script('AT_classA_kmer.py'),
                '-i', args.input,
                '-t', str(args.threads),
                '--top-k',         str(args.top_k),
                '--window-size',   str(args.window_size),
                '--pctsim',        str(args.pctsim),
                '--outdir',        proc_dir,
                '--classify-only',
            ]
            _run(cmd_a_classify, 'Class A — classify (kmer)')

            # Phase 2 — run A download+build concurrently with full Class B
            cmd_a_dl = [
                py_main, _script('AT_classA_kmer.py'),
                '-i', args.input,
                '-t', str(args.threads),
                '--top-k',              str(args.top_k),
                '--window-size',        str(args.window_size),
                '--pctsim',             str(args.pctsim),
                '--outdir',             proc_dir,
                '--download-build-only',
            ]

            if not args.skip_classB:
                cmd_b = [
                    py_main, _script('AT_classB.py'),
                    '-i', proc_dir,
                    '-t', str(args.threads),
                    '--max-indels',    str(args.max_indels),
                    '--max-indel-len', str(args.max_indel_len),
                    '--min-pctsim',    str(b_min_pctsim),
                    '--mm-iters',      str(args.mm_iters),
                    '--ccd-iters',     str(args.ccd_iters),
                    '--ccd-tol',       str(args.ccd_tol),
                    '--flank',         str(args.flank),
                    '--loop-closer',   args.loop_closer,
                    *_flag('--promod3-data-dir', args.promod3_data_dir),
                    *_flag('--limit', args.b_limit),
                ]
                print()
                print('=' * 60)
                print('  AlphaTracer  ►  Class A download + Class B  [concurrent]')
                print('=' * 60)

                def _run_silent(cmd, label):
                    env = {**os.environ}
                    result = subprocess.run(cmd, env=env)
                    if result.returncode != 0:
                        sys.exit(f'\n[FATAL] {label} exited with code {result.returncode}. Aborting.')

                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_a = ex.submit(_run_silent, cmd_a_dl, 'Class A download+build')
                    fut_b = ex.submit(_run_silent, cmd_b, 'Class B')
                    for fut in as_completed([fut_a, fut_b]):
                        fut.result()  # re-raise any exception
                print('  [concurrent phase complete]')
            else:
                _run(cmd_a_dl, 'Class A — download+build (kmer)')
                print('[SKIP] Class B')
    else:
        print(f'[SKIP] Class A  (using existing {proc_dir}/)')
        # ── Class B (standalone, A was skipped) ───────────────────────────────
        if not args.skip_classB:
            cmd_b = [
                py_main, _script('AT_classB.py'),
                '-i', proc_dir,
                '-t', str(args.threads),
                '--max-indels',    str(args.max_indels),
                '--max-indel-len', str(args.max_indel_len),
                '--min-pctsim',    str(b_min_pctsim),
                '--mm-iters',      str(args.mm_iters),
                '--ccd-iters',     str(args.ccd_iters),
                '--ccd-tol',       str(args.ccd_tol),
                '--flank',         str(args.flank),
                '--loop-closer',   args.loop_closer,
                *_flag('--promod3-data-dir', args.promod3_data_dir),
                *_flag('--limit', args.b_limit),
            ]
            _run(cmd_b, 'Class B')
        else:
            print('[SKIP] Class B')

    # ── Class C + D ───────────────────────────────────────────────────────────
    if not args.skip_classC:
        cmd_cd = [
            py_cd, _script('AT_classC_and_D.py'),
            '-i', proc_dir,
            '-t', str(args.threads),
            '--min-pctsim',             str(c_min_pctsim),
            '--window-size',            str(args.window_size),
            '--pae-cutoff',             str(args.pae_cutoff),
            '--pae-power',              str(args.pae_power),
            '--pae-resolution',         str(args.pae_resolution),
            '--pae-resolution-large',   str(args.pae_resolution_large),
            '--large-domain-threshold', str(args.large_domain_threshold),
            '--mm-iters',               str(args.mm_iters_C),
            '--ccd-iters',              str(args.ccd_iters),
            '--ccd-tol',                str(args.ccd_tol),
            '--flank',                  str(args.flank),
            '--loop-closer',            args.loop_closer,
            '--backend',                args.backend,
            *_flag('--promod3-data-dir', args.promod3_data_dir),
            '--min-frag-len',           str(args.min_frag_len),
            '--lbfgsb-iters',           str(args.lbfgsb_iters),
            '--mini-model-size',        args.mini_model_size,
            '--anchor-k',               str(args.anchor_k),
            '--plddt-threshold',        str(args.plddt_threshold),
            '--max-recyclings',         str(args.max_recyclings),
            '--ppl-short-threshold',    str(args.ppl_short_threshold),
            '--batch-tokens',           str(args.batch_tokens),
            '--max-seq-len',            str(args.max_seq_len),
            *_flag('--limit',        args.c_limit),
            *_flag('--classD-limit', args.classD_limit),
            *_bool_flag('--fill-missing', args.fill_missing),
            *_bool_flag('--no-classD',    args.no_classD),
        ]
        _run(cmd_cd, 'Class C + D', extra_env={'KMP_DUPLICATE_LIB_OK': 'TRUE'})
    else:
        print('[SKIP] Class C/D')

    print()
    print('=' * 60)
    print('AlphaTracer  —  Pipeline complete.')
    print(f'  Output directory: {proc_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
