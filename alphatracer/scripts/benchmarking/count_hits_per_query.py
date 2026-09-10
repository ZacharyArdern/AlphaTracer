#!/usr/bin/env python3
"""Count diamond hits per query, fully streaming. Run from AT_DBs/."""
import polars as pl

(
    pl.scan_csv('afdb_hits_merged.tsv', separator='\t', has_header=False,
                new_columns=['query', 'target', 'pident', 'length', 'qlen', 'evalue'],
                schema_overrides={'pident': pl.Float32, 'length': pl.Int32, 'evalue': pl.Float64})
    .group_by('query')
    .agg(pl.len().alias('n_hits'))
    .sort('n_hits', descending=True)
    .sink_csv('hits_per_query.tsv', separator='\t')
)

df = pl.read_csv('hits_per_query.tsv', separator='\t')
print(f'Total queries: {df.height:,}')
print(df['n_hits'].describe())
print('\nDistribution:')
for t in [1, 10, 50, 100, 500, 1000, 5000, 10000]:
    n = df.filter(pl.col('n_hits') < t).height
    print(f'  <{t:6d} hits: {n:,} ({100*n/df.height:.1f}%)')
