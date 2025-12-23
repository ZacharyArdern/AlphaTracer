#! /usr/bin/env python3

import os 
import subprocess
import polars as pl  
import argparse
from Bio import SeqIO
from pathlib import Path
import plotext as plt
import pycurl
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from collections import Counter
import time 

# example: python3 alpha_search_v0.1.py -i GCF_000621645.1_500.fa -d afdb50.dmnd -t 6

parser = argparse.ArgumentParser(description="Search query sequences against a database of structure models for homologs")

parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input fasta file of query sequences')

parser.add_argument('-d', '--database', type=str, default='AFDB.dmnd',
                        help='path to pre-created diamond database of sequences in AlphaFoldDB')

parser.add_argument('-t', '--threads', type=int, default=4,
                        help='CPU threads to use')

args = parser.parse_args()




pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(3)
pl.Config.set_fmt_str_lengths(25)
pl.Config.set_tbl_hide_dataframe_shape(True)

ascii_art1 = r"""
     _    _     ____  _   _    _  _____ ____      _    ____ _____ ____  
    / \  | |   |  _ \| | | |  / \|_   _|  _ \    / \  / ___| ____|  _ \ 
   / _ \ | |   | |_) | |_| | / _ \ | | | |_) |  / _ \| |   |  _| | |_) |
  / ___ \| |___|  __/|  _  |/ ___ \| | |  _ <  / ___ \ |___| |___|  _ < 
 /_/   \_\_____|_|   |_| |_/_/   \_\_| |_| \_\/_/   \_\____|_____|_| \_\
"""

ascii_art2 = r"""
        _       _                                         _             
   __ _| |_ __ | |__   __ _       ___  ___  __ _ _ __ ___| |__          
  / _` | | '_ \| '_ \ / _` |_____/ __|/ _ \/ _` | '__/ __| '_ \         
 | (_| | | |_) | | | | (_| |_____\__ \  __/ (_| | | | (__| | | |        
  \__,_|_| .__/|_| |_|\__,_|     |___/\___|\__,_|_|  \___|_| |_|        
         |_|                                                                          
"""
# ASCII art from: https://patorjk.com/software/taag/#p=display&f=Ivrit&t=ALPHATRACER%0Aalpha-search

plt.colorize(f"{ascii_art1}\n© ZACHARY ARDERN 2025\ncomments to: z.ardern@gmail.com or https://github.com/ZacharyArdern/AlphaTracer/", "blue", "bold", "-", True)

print( 
    "\n\n",
    "RUNNING MODULE 1; SEARCH FOR AFDB HOMOLOGS (DIAMOND) AND DOWNLOAD PDB STRUCTURES",
    f"{ascii_art2}" + "\n" * 2,
    sep="",
)


# FILTER PROTEINS 
# EXCLUDE THOSE WHICH CONTAIN NON-CANONICAL AMINO ACID SYMBOL "X" OR ARE >1000 AMINO ACIDS:


def filter_fasta(input_path, output_path):
    filtered = []
    for record in SeqIO.parse(input_path, 'fasta'):
        if 'X' not in str(record.seq) and len(record.seq) <= 2000:
        # if 'X' not in str(record.seq):
            filtered.append(f'>{record.id}\n{record.seq}')
    with open(output_path, 'w') as out_f:
        out_f.write('\n'.join(filtered))


input_basename = os.path.basename(args.input).split('.')[0]
outdir = f"AT_processing_{input_basename}"

# Make the output directory
os.makedirs(outdir, exist_ok=True)

blast_output_file = os.path.abspath(os.path.join(outdir, 'tmp_blast.txt'))

# note: sometimes the initial diamond blastp output fails unless filename is used instead of path var in the cmd below
# this is (it appears) a memory issue, fixed by using a smaller value for "b" (was previously using 4)

filtered_fasta = os.path.join(outdir, 'input_seqs_filtered.fa')

# note: the comp-based-stats option and no [repeat] masking are used to avoid adding Xs to full_qseq
# a faster approach, if needed, would be to include masking and then retrieve the q_seq from the input file and s_seq from the DB 

# Construct the command
cmd = [
    "diamond", "blastp",
    "--fast",
    "-q", filtered_fasta,
    "--db", args.database,
    "-b", "2",
    "-o", blast_output_file,
    "--threads", str(args.threads),
    "--comp-based-stats", "2", 
    "--masking", "0",
    "-f", "6", "approx_pident", "sseqid", "qseqid", "evalue", "slen", "qlen", "full_qseq", "full_sseq",
    "--evalue", "1e-5",
    "--max-target-seqs", "5"
]

# Run the commands

print("Output path:", blast_output_file)
print("Output directory exists:", os.path.isdir(os.path.dirname(blast_output_file)))
print("Can write to output directory:", os.access(os.path.dirname(blast_output_file), os.W_OK), '\n\n')

filter_fasta(args.input, filtered_fasta)


# issues with uncompleted filtered_fasta seem to be solved by adding this code (3 lines):
while not (os.path.exists(filtered_fasta) and os.path.getsize(filtered_fasta) > 0):
    print("Waiting for filtered FASTA to be written...")
    time.sleep(0.5)


result = subprocess.run(cmd)
if result.returncode != 0:
    raise RuntimeError(f"DIAMOND failed with exit code {result.returncode}")

while not (os.path.exists(blast_output_file) and os.path.getsize(blast_output_file) > 0):
    print("Waiting for DIAMOND 'BLAST' output to be written...")
    time.sleep(0.5)

# GET TOP HITS

new_cols_list = ['approx_pident', 'sseqid', 'qseqid', 'evalue', 'slen', 'qlen', 'full_qseq', 'full_sseq']
afdbhits_df = pl.read_csv(blast_output_file, has_header=False, separator='\t', new_columns=new_cols_list)
afdbhits_df.write_parquet(os.path.join(outdir, 'afdb_allhits.pq'))


afdbtophits_df = afdbhits_df.group_by(['qseqid'], maintain_order=True).agg(pl.all().first()).filter(pl.col('approx_pident') >= 30)
# afdbtophits_df = afdbhits_df.filter(pl.col('approx_pident') >= 30).group_by(['qseqid'], maintain_order=True).agg(pl.all().sort_by('slen', descending=True).first())
print('HITS DATA FRAME', '\n', afdbtophits_df, sep='')

afdbtophits_df.write_csv(os.path.join(outdir, 'afdbtophits.txt'), include_header=False, separator='\t')

afdbtophits_df.write_parquet(os.path.join(outdir, 'afdb_tophits.pq'))



# DOWNLOAD CORRESPONDING ALPHAFOLD MODELS
# need to update to parallel download with e.g. curl or rsync

# cat AT_processing/afdbtophits.txt | 
# awk -F "\t" '{split($3,a,":"); print "https://alphafold.ebi.ac.uk/files/"a[2]"-model_v4.pdb"}' > AT_processing/urls.txt ; 

urls = []
with open(os.path.join(outdir, 'afdbtophits.txt')) as f:
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) >= 3 and ':' in fields[2]:
            acc = fields[2].split(':')[1]
            urls.append(f'https://alphafold.ebi.ac.uk/files/{acc}-model_v4.pdb')
        elif 'AF' in fields[2]:
            acc = fields[2]
            urls.append(f'https://alphafold.ebi.ac.uk/files/{acc}.pdb')
            # urls.append(f'https://alphafold.ebi.ac.uk/files/{acc}.bcif')

                        
print('\n', 'EXAMPLE URLS: ', '\n', urls[0:10], '\n', sep='')

urls_unique = list(set(urls))

# test ! -d mkdir AT_processing/AF_pdbs ; 
# wget -nc -i AT_processing/urls.txt -P AT_processing/AF_pdbs ; 

# note, downloading bcifs is ~2x faster than downloading pdb


# pdb_dir = './AT_processing/AF_bcifs/'
pdb_dir = os.path.join(outdir, 'AF_pdbs/')
os.makedirs(pdb_dir, exist_ok=True)

print('\n', 'DOWNLOADING AF FILES', '\n', sep='')


def download(url):
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        return f'Skipped (no filename): {url}'

    path = os.path.join(pdb_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return f'Skipped (already exists): {filename}'

    try:
        with open(path, 'wb') as f:
            c = pycurl.Curl()
            c.setopt(c.URL, url)
            c.setopt(c.WRITEDATA, f)
            c.perform()
            c.close()
        return f'Downloaded: {filename}'
    except Exception as e:
        return f'Failed: {url} ({e})'

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(download, urls_unique))

skipped_count = 0
max_skipped_to_show = 5
summary = Counter()
skipped_printed_etc = False

for r in results:
    if r.startswith('Skipped (already exists):'):
        summary['skipped'] += 1
        if skipped_count < max_skipped_to_show:
            print(r)
        elif not skipped_printed_etc:
            print('...etc')
            skipped_printed_etc = True
        skipped_count += 1
    elif r.startswith('Downloaded:'):
        summary['success'] += 1
        print(r)
    elif r.startswith('Skipped (no filename):'):
        summary['no_filename'] += 1
        print(r)
    elif r.startswith('Failed:'):
        summary['failed'] += 1
        print(r)
    else:
        summary['other'] += 1
        print(r)

print('\nSummary:')
for key in ['success', 'skipped', 'no_filename', 'failed', 'other']:
    if summary[key]:
        print(f'{key.capitalize()} (downloaded now or previously): {summary[key]}')


print(
    "\n",
    "COMPLETED MODULE 1; FOUND AND DOWNLOADED HOMOLOGOUS PROTEIN STRUCTURE MODELS.",
    "\n",
    sep="\n",
)