#!/usr/bin/env python3
"""
Pick one random species-representative genome per GTDB phylum.
Reproducible via fixed seed. Outputs a TSV with accession, phylum, and domain.

Optionally extracts proteomes from the GTDB protein tar.gz:
  gtdb_proteins_aa_reps_r232.tar.gz
  └── protein_faa_reps/{bacteria|archaea}/{accession}_protein.faa.gz

Usage:
    python sample_one_per_phylum.py [--seed 42] [--out sampled_reps.tsv]
    python sample_one_per_phylum.py --tar gtdb_proteins_aa_reps_r232.tar.gz --extract-to proteomes/
"""
import gzip
import csv
import random
import argparse
import tarfile
import shutil
from pathlib import Path
from collections import defaultdict

DEFAULT_GTDB_DIR = Path.cwd()


def extract_phylum(taxonomy: str) -> str:
    for part in taxonomy.split(';'):
        part = part.strip()
        if part.startswith('p__'):
            return part
    return ''


def load_reps(path: Path, domain: str) -> dict[str, list[tuple[str, str]]]:
    """Return dict: phylum -> list of (accession, domain) for GTDB reps."""
    phylum_to_accs = defaultdict(list)
    with gzip.open(path, 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['gtdb_representative'] != 't':
                continue
            phylum = extract_phylum(row['gtdb_taxonomy'])
            if not phylum or phylum == 'p__':
                continue
            acc = row['accession']
            phylum_to_accs[phylum].append((acc, domain))
    return phylum_to_accs


def tar_path(accession: str, domain: str) -> str:
    """Return the expected member path inside the GTDB proteome tar.gz."""
    subdir = 'archaea' if domain == 'Archaea' else 'bacteria'
    return f'protein_faa_reps/{subdir}/{accession}_protein.faa.gz'


def extract_proteomes(sampled: list, tar_gz: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {tar_path(acc, domain): (phylum, acc) for phylum, domain, acc in sampled}

    found = 0
    print(f'Scanning {tar_gz.name} for {len(wanted)} proteomes...')
    with tarfile.open(tar_gz, 'r:gz') as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            phylum, acc = wanted[member.name]
            dest = out_dir / f'{acc}_protein.faa.gz'
            if dest.exists():
                found += 1
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(dest, 'wb') as f:
                shutil.copyfileobj(src, f)
            found += 1
            if found % 20 == 0:
                print(f'  {found}/{len(wanted)} extracted...')

    missing = len(wanted) - found
    print(f'Extracted {found}/{len(wanted)} proteomes to {out_dir}')
    if missing:
        print(f'  WARNING: {missing} accessions not found in tar — check accession format')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gtdb-dir', type=Path, default=DEFAULT_GTDB_DIR,
                        help='Directory containing bac120_metadata_r232.tsv.gz and ar53_metadata_r232.tsv.gz')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--out', default='sampled_reps.tsv', help='Output TSV path')
    parser.add_argument('--tar', type=Path, default=None,
                        help='Path to gtdb_proteins_aa_reps_r232.tar.gz for proteome extraction')
    parser.add_argument('--extract-to', type=Path, default=Path('rep_proteomes'),
                        help='Directory to extract proteome .faa.gz files into (default: rep_proteomes/)')
    args = parser.parse_args()

    files = [
        (args.gtdb_dir / 'bac120_metadata_r232.tsv.gz', 'Bacteria'),
        (args.gtdb_dir / 'ar53_metadata_r232.tsv.gz',   'Archaea'),
    ]

    rng = random.Random(args.seed)

    all_phyla: dict[str, list[tuple[str, str]]] = {}
    for path, domain in files:
        for phylum, accs in load_reps(path, domain).items():
            all_phyla.setdefault(phylum, []).extend(accs)

    out_path = Path(args.out)
    sampled = []
    for phylum in sorted(all_phyla):
        candidates = sorted(all_phyla[phylum])  # sort before rng for determinism
        chosen_acc, domain = rng.choice(candidates)
        sampled.append((phylum, domain, chosen_acc))

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['phylum', 'domain', 'accession'])
        writer.writerows(sampled)

    bac = sum(1 for _, d, _ in sampled if d == 'Bacteria')
    arc = sum(1 for _, d, _ in sampled if d == 'Archaea')
    print(f'Sampled {len(sampled)} genomes: {bac} bacteria, {arc} archaea')
    print(f'Output: {out_path}')

    if args.tar:
        if not args.tar.exists():
            print(f'ERROR: tar file not found: {args.tar}')
            return
        extract_proteomes(sampled, args.tar, args.extract_to)


if __name__ == '__main__':
    main()
