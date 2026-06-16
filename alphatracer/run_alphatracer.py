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

  # Show full verbose output instead of progress bar:
  python run_alphatracer.py -i proteins.fasta --verbose
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).parent          # alphatracer/
_PKG_PARENT = str(_HERE.parent)              # repo root — added to PYTHONPATH


def _script(name: str) -> str:
    return str(_HERE / name)


def _make_env(extra: dict | None = None) -> dict:
    env = {**os.environ, **(extra or {})}
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = _PKG_PARENT + (os.pathsep + existing if existing else '')
    return env


def _run(cmd: list[str], label: str, extra_env: dict | None = None) -> None:
    """Verbose mode: print header + stream subprocess output."""
    print()
    print('=' * 60)
    print(f'  AlphaTracer  ►  {label}')
    print('  Command: ' + ' '.join(cmd))
    print('=' * 60)
    result = subprocess.run(cmd, env=_make_env(extra_env))
    if result.returncode != 0:
        sys.exit(f'\n[FATAL] {label} exited with code {result.returncode}. Aborting.')


def _run_quiet(cmd: list[str], label: str, log_path: str,
               extra_env: dict | None = None) -> None:
    """Quiet mode: capture output to log file, raise on failure."""
    with open(log_path, 'a') as lf:
        lf.write(f'\n{"=" * 60}\n  {label}\n{"=" * 60}\n')
        lf.write('  Command: ' + ' '.join(cmd) + '\n\n')
        result = subprocess.run(cmd, env=_make_env(extra_env), stdout=lf, stderr=lf)
    if result.returncode != 0:
        sys.stdout.write('\n')
        sys.exit(
            f'\n[FATAL] {label} exited with code {result.returncode}.\n'
            f'        See {log_path} for details. Aborting.'
        )


def _flag(name: str, value) -> list[str]:
    return [name, str(value)] if value else []


def _bool_flag(name: str, enabled: bool) -> list[str]:
    return [name] if enabled else []


# ── Progress bar ───────────────────────────────────────────────────────────────

def _count_pdbs(path: str) -> int:
    try:
        return sum(1 for e in os.scandir(path) if e.name.endswith('.pdb'))
    except (FileNotFoundError, NotADirectoryError):
        return 0


def _count_fasta(path: str) -> int:
    try:
        open_fn = gzip.open if path.endswith('.gz') else open
        with open_fn(path, 'rt') as f:
            return sum(1 for line in f if line.startswith('>'))
    except Exception:
        return 0


class _StatusBar:
    """Single updating status line shown during quiet-mode execution."""

    _BAR_WIDTH = 28

    def __init__(self, proc_dir: str, total: int, log_path: str):
        self.proc_dir  = proc_dir
        self.total     = total
        self.log_path  = log_path
        self._phase    = 'Starting...'
        self._stop     = threading.Event()
        self._lock     = threading.Lock()   # guards _phase + stdout writes
        self._thread   = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> '_StatusBar':
        self._thread.start()
        return self

    def phase(self, label: str) -> None:
        with self._lock:
            if self._phase == label:
                return
            self._phase = label
            # Clear the current \r bar line, then print the phase as a permanent line.
            # Hold _lock while writing so _draw() cannot interleave.
            sys.stdout.write(f'\x1b[2K\r  {label}\n')
            sys.stdout.flush()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._draw(final=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _read_total_file(self, cls: str) -> int | None:
        """Read .classX_total written by the subprocess once its candidate count is known."""
        try:
            with open(os.path.join(self.proc_dir, f'.class{cls}_total')) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _class_A_total(self) -> int | None:
        """Class A total from classA.pq written after kmer-classify phase."""
        try:
            import polars as pl
            pq = os.path.join(self.proc_dir, 'classA.pq')
            return pl.read_parquet(pq)['qseqid'].n_unique()
        except Exception:
            return None

    def _class_counts(self) -> dict[str, tuple[int, int | None]]:
        """Returns {cls: (done, total_or_None)} for each class."""
        result = {}
        for cls in ('A', 'B', 'C', 'D'):
            done = _count_pdbs(os.path.join(self.proc_dir, f'output_pdbs_class{cls}'))
            if cls == 'A':
                total = self._class_A_total()
            else:
                total = self._read_total_file(cls)
            result[cls] = (done, total)
        return result

    def _read_db_counts(self, cls: str) -> dict | None:
        """Read .classX_db_counts JSON written live by AT_classA/B scripts."""
        try:
            import json
            with open(os.path.join(self.proc_dir, f'.class{cls}_db_counts')) as f:
                return json.load(f)
        except Exception:
            return None

    def _read_cd_status(self) -> str | None:
        """Read subprocess status written by AT_classC_and_D.py to .cd_status."""
        try:
            with open(os.path.join(self.proc_dir, '.cd_status')) as f:
                return f.read().strip() or None
        except Exception:
            return None

    def _draw(self, final: bool = False) -> None:
        counts = self._class_counts()
        done   = sum(d for d, _ in counts.values())
        total  = self.total
        pct    = done / max(total, 1)
        w      = self._BAR_WIDTH
        filled = min(int(w * pct), w)
        bar    = '█' * filled + '░' * (w - filled)

        active = [(k, d, t) for k, (d, t) in counts.items() if d > 0]
        if active:
            parts = []
            for k, d, t in active:
                base = f'Class {k}: {d}/{t}' if t is not None else f'Class {k}: {d}'
                if k in ('A', 'B'):
                    db = self._read_db_counts(k)
                    if db and db.get('ok'):
                        ok = db['ok']
                        afdb = ok.get('afdb', 0)
                        esm  = ok.get('esm_atlas', 0)
                        base += f' (AF={afdb} ESM={esm})'
                parts.append(base)
            detail = '  '.join(parts)
        else:
            with self._lock:
                detail = self._phase

        if final:
            line = f'\x1b[2K\r  [{bar}] {done}/{total} seqs ({100*pct:.0f}%)  |  Done\n'
        else:
            cols = shutil.get_terminal_size((80, 24)).columns
            content = f'  [{bar}] {done}/{total} seqs ({100*pct:.0f}%)  |  {detail}'
            line = f'\x1b[2K\r{content[:cols - 1]}'

        with self._lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            cd = self._read_cd_status()
            if cd:
                self.phase(cd)
            self._draw()


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
    p.add_argument(
        '--dbdir', default=None, metavar='DIR',
        help='Directory containing AFDB parquet and sketch index '
             '(default: current working directory)',
    )

    # ── Shared ────────────────────────────────────────────────────────────────
    shared = p.add_argument_group('shared options (forwarded to all relevant scripts)')
    shared.add_argument(
        '--diamond', action='store_true', default=False,
        help='Use DIAMOND blastp for Class A search instead of kmer sketch (default: kmer)',
    )
    shared.add_argument(
        '-d', '--database', default=None,
        help='DIAMOND database for Class A (only used with --diamond; '
             'default: bacarc8080.dmnd in --dbdir)',
    )
    shared.add_argument('--top-k', type=int, default=5,
                        help='Hits per query from kmer search (default: 5)')
    shared.add_argument('--sketch-db', default=None, metavar='PATH',
                        help='Custom sketch database parquet for kmer Class A search '
                             '(e.g. ESMAtlas). Forwarded to AT_classA_kmer.py.')
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
                       help='Max indels (strict path)')
    grp_b.add_argument('--max-indel-len', type=int, default=5,
                       help='Max length per indel (strict path)')
    grp_b.add_argument('--max-loop-indels', type=int, default=8,
                       help='Max indels when all fall in loop/coil context (default: 8)')
    grp_b.add_argument('--max-loop-indel-len', type=int, default=20,
                       help='Max indel length in loop/coil context (default: 20)')
    grp_b.add_argument('--b-min-pctsim', type=float, default=40.0,
                       help='Min identity for Class B candidates')
    grp_b.add_argument('--b-limit', type=int, default=0,
                       help='Process only first N Class B structures (0=all)')

    # ── Class C/D ─────────────────────────────────────────────────────────────
    grp_cd = p.add_argument_group('Class C/D options')
    grp_cd.add_argument('--c-min-pctsim', type=float, default=40.0,
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
    grp_cd.add_argument('--min-domain-size', type=int, default=30,
                        help='Minimum domain residue count for Class C (default: 30)')
    grp_cd.add_argument('--min-domain-plddt', type=float, default=70.0,
                        help='Minimum mean AF pLDDT over domain for Class C (default: 70)')
    grp_cd.add_argument('--c-limit', type=int, default=0,
                        help='Limit Class C to first N sequences (0=all)')
    grp_cd.add_argument('--no-fill-missing', action='store_false', dest='fill_missing', default=True,
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
    grp_cd.add_argument('--min-recycle-plddt', type=float, default=50.0,
                        help='Skip recycling if round-0 pLDDT < this — likely disordered (default: 50)')
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
    ctrl.add_argument('--outdir', default=None, metavar='DIR',
                      help='Processing directory to use or resume '
                           '(default: AT_processing_{input stem} in current directory)')
    ctrl.add_argument('--verbose', action='store_true', default=False,
                      help='Print full output from each stage instead of a progress bar')

    return p.parse_args()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve dbdir and propagate to all subprocesses via AT_AFDB_DIR.
    dbdir = os.path.abspath(args.dbdir) if args.dbdir else os.getcwd()
    os.environ['AT_AFDB_DIR'] = dbdir

    # Resolve DIAMOND database path (only used with --diamond).
    if args.database is None:
        args.database = os.path.join(dbdir, 'bacarc8080.dmnd')

    input_path   = Path(args.input)
    proc_dir     = os.path.abspath(args.outdir) if args.outdir else f'AT_processing_{input_path.stem}'
    py_main      = args.python
    py_cd        = args.classcd_python or py_main
    verbose      = args.verbose
    log_path     = os.path.join(proc_dir, 'alphatracer.log')

    total_seqs   = _count_fasta(args.input)

    print('=' * 60)
    print('AlphaTracer  —  Full Pipeline Wrapper')
    print('=' * 60)
    search_method = 'diamond' if args.diamond else 'kmer'
    # kmer approx_pident is Jaccard-based and systematically lower than NW
    # identity; pass threshold=1 to the pre-NW filter and let Class B/C apply
    # the real threshold after NW alignment instead.
    b_min_pctsim  = args.b_min_pctsim if args.diamond else 1
    c_min_pctsim  = args.c_min_pctsim if args.diamond else 1

    print(f'  Input:        {args.input}  ({total_seqs} sequences)')
    print(f'  DB dir:       {dbdir}')
    print(f'  Processing:   {proc_dir}/')
    print(f'  Search:       {search_method}')
    print(f'  Python (A/B): {py_main}')
    print(f'  Python (C/D): {py_cd}')
    print(f'  Loop closer:  {args.loop_closer}')
    print(f'  Backend:      {args.backend}')
    print(f'  Min identity: Class A ≥{args.pctsim}% (window={args.window_size} aa)  '
          f'B ≥{b_min_pctsim}%  C ≥{c_min_pctsim}%')
    skips = [c for c, s in [('A', args.skip_classA), ('B', args.skip_classB),
                              ('C', args.skip_classC)] if s]
    skips += (['D'] if args.no_classD else [])
    print(f'  Skip:         {", ".join(skips) or "none"}')
    if not verbose:
        print(f'  Log:          {log_path}')
    print()

    # ── ProMod3 database setup ─────────────────────────────────────────────────
    if args.loop_closer == 'promod3':
        from setup_databases import ensure_promod3_databases, default_data_dir
        data_dir = ensure_promod3_databases(
            data_dir=args.promod3_data_dir, verbose=True)
        os.environ['PROMOD3_SHARED_DATA_PATH'] = str(data_dir)

    # ── Set up quiet-mode progress bar ────────────────────────────────────────
    bar: _StatusBar | None = None
    if not verbose:
        os.makedirs(proc_dir, exist_ok=True)
        bar = _StatusBar(proc_dir, total_seqs, log_path).start()

    def run(cmd, label, extra_env=None):
        if verbose:
            _run(cmd, label, extra_env)
        else:
            _run_quiet(cmd, label, log_path, extra_env)

    # ── Class A ───────────────────────────────────────────────────────────────
    if not args.skip_classA:
        if args.diamond:
            if bar:
                bar.phase('Searching for Class A sequences (DIAMOND)...')
            cmd_a = [
                py_main, _script('AT_classA.py'),
                '-i', args.input,
                '-d', args.database,
                '-t', str(args.threads),
                '--window-size', str(args.window_size),
                '--pctsim',      str(args.pctsim),
            ]
            run(cmd_a, f'Class A ({search_method})')

            if not args.skip_classB:
                if bar:
                    bar.phase('Building Class B structures...')
                cmd_b = [
                    py_main, _script('AT_classB.py'),
                    '-i', proc_dir,
                    '-t', str(args.threads),
                    '--max-indels',         str(args.max_indels),
                    '--max-indel-len',      str(args.max_indel_len),
                    '--max-loop-indels',    str(args.max_loop_indels),
                    '--max-loop-indel-len', str(args.max_loop_indel_len),
                    '--min-pctsim',         str(b_min_pctsim),
                    '--mm-iters',      str(args.mm_iters),
                    '--ccd-iters',     str(args.ccd_iters),
                    '--ccd-tol',       str(args.ccd_tol),
                    '--flank',         str(args.flank),
                    '--loop-closer',   args.loop_closer,
                    *_flag('--promod3-data-dir', args.promod3_data_dir),
                    *_flag('--limit', args.b_limit),
                ]
                run(cmd_b, 'Class B')
            else:
                if verbose:
                    print('[SKIP] Class B')
        else:
            # kmer: Phase 1 — classify only (~10s), writes classA.pq
            if bar:
                bar.phase('Searching for Class A sequences (kmer)...')
            cmd_a_classify = [
                py_main, _script('AT_classA_kmer.py'),
                '-i', args.input,
                '-t', str(args.threads),
                '--top-k',         str(args.top_k),
                '--window-size',   str(args.window_size),
                '--pctsim',        str(args.pctsim),
                '--outdir',        proc_dir,
                '--classify-only',
                *_flag('--sketch-db', args.sketch_db),
            ]
            run(cmd_a_classify, 'Class A — classify (kmer)')

            # Phase 2 — A download+build concurrently with full Class B
            cmd_a_dl = [
                py_main, _script('AT_classA_kmer.py'),
                '-i', args.input,
                '-t', str(args.threads),
                '--top-k',              str(args.top_k),
                '--window-size',        str(args.window_size),
                '--pctsim',             str(args.pctsim),
                '--outdir',             proc_dir,
                '--download-build-only',
                *_flag('--sketch-db', args.sketch_db),
            ]

            if not args.skip_classB:
                cmd_b = [
                    py_main, _script('AT_classB.py'),
                    '-i', proc_dir,
                    '-t', str(args.threads),
                    '--max-indels',         str(args.max_indels),
                    '--max-indel-len',      str(args.max_indel_len),
                    '--max-loop-indels',    str(args.max_loop_indels),
                    '--max-loop-indel-len', str(args.max_loop_indel_len),
                    '--min-pctsim',         str(b_min_pctsim),
                    '--mm-iters',      str(args.mm_iters),
                    '--ccd-iters',     str(args.ccd_iters),
                    '--ccd-tol',       str(args.ccd_tol),
                    '--flank',         str(args.flank),
                    '--loop-closer',   args.loop_closer,
                    *_flag('--promod3-data-dir', args.promod3_data_dir),
                    *_flag('--limit', args.b_limit),
                ]

                if verbose:
                    print()
                    print('=' * 60)
                    print('  AlphaTracer  ►  Class A download + Class B  [concurrent]')
                    print('=' * 60)
                else:
                    bar.phase('Building Class A and Class B structures...')

                def _run_sub(cmd, label):
                    env = _make_env()
                    if verbose:
                        result = subprocess.run(cmd, env=env)
                    else:
                        with open(log_path, 'a') as lf:
                            lf.write(f'\n=== {label} ===\n')
                            result = subprocess.run(cmd, env=env, stdout=lf, stderr=lf)
                    if result.returncode != 0:
                        if not verbose:
                            sys.stdout.write('\n')
                        msg = f'\n[FATAL] {label} exited with code {result.returncode}.'
                        if not verbose:
                            msg += f'\n        See {log_path} for details.'
                        sys.exit(msg + ' Aborting.')

                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_a = ex.submit(_run_sub, cmd_a_dl, 'Class A download+build')
                    fut_b = ex.submit(_run_sub, cmd_b, 'Class B')
                    for fut in as_completed([fut_a, fut_b]):
                        fut.result()

                if verbose:
                    print('  [concurrent phase complete]')
            else:
                if bar:
                    bar.phase('Building Class A structures...')
                run(cmd_a_dl, 'Class A — download+build (kmer)')
                if verbose:
                    print('[SKIP] Class B')
    else:
        if verbose:
            print(f'[SKIP] Class A  (using existing {proc_dir}/)')
        if not args.skip_classB:
            if bar:
                bar.phase('Building Class B structures...')
            cmd_b = [
                py_main, _script('AT_classB.py'),
                '-i', proc_dir,
                '-t', str(args.threads),
                '--max-indels',         str(args.max_indels),
                '--max-indel-len',      str(args.max_indel_len),
                '--max-loop-indels',    str(args.max_loop_indels),
                '--max-loop-indel-len', str(args.max_loop_indel_len),
                '--min-pctsim',         str(b_min_pctsim),
                '--mm-iters',      str(args.mm_iters),
                '--ccd-iters',     str(args.ccd_iters),
                '--ccd-tol',       str(args.ccd_tol),
                '--flank',         str(args.flank),
                '--loop-closer',   args.loop_closer,
                *_flag('--promod3-data-dir', args.promod3_data_dir),
                *_flag('--limit', args.b_limit),
            ]
            run(cmd_b, 'Class B')
        else:
            if verbose:
                print('[SKIP] Class B')

    # ── Class C + D ───────────────────────────────────────────────────────────
    if not args.skip_classC:
        # Remove stale sentinel files from any previous run so the progress bar
        # does not show leftover counts before the current run writes them.
        for _stale in ('.cd_status', '.classC_total', '.classD_total'):
            try:
                os.remove(os.path.join(proc_dir, _stale))
            except FileNotFoundError:
                pass
        if bar:
            bar.phase('Building Class C and D structures...')
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
            '--min-domain-size',        str(args.min_domain_size),
            '--min-domain-plddt',       str(args.min_domain_plddt),
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
            '--min-recycle-plddt',      str(args.min_recycle_plddt),
            '--batch-tokens',           str(args.batch_tokens),
            '--max-seq-len',            str(args.max_seq_len),
            *_flag('--limit',        args.c_limit),
            *_flag('--classD-limit', args.classD_limit),
            *_bool_flag('--no-fill-missing', not args.fill_missing),
            *_bool_flag('--no-classD',    args.no_classD),
        ]
        run(cmd_cd, 'Class C + D', extra_env={'KMP_DUPLICATE_LIB_OK': 'TRUE'})
    else:
        if verbose:
            print('[SKIP] Class C/D')

    # ── Done ──────────────────────────────────────────────────────────────────
    if bar:
        bar.stop()

    print()
    print('=' * 60)
    print('AlphaTracer  —  Pipeline complete.')
    print(f'  Output directory: {proc_dir}/')
    if not verbose:
        print(f'  Full log:         {log_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
