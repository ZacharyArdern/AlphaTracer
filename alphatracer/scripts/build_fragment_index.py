#!/usr/bin/env python3
"""
Build a local protein_hash -> fragment_id index for the ESM Atlas Lance dataset.
Scans all fragments in parallel (16 threads), writes chunked parquets, then
merges into a sorted DuckDB database for fast batch lookups.
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import lancedb
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import time
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Suppress the noisy 403 manifest warnings from lancedb
logging.getLogger('lance').setLevel(logging.ERROR)
os.environ['RUST_LOG'] = 'error'

OUT_DIR = Path(__file__).parent / 'folds_1B_index'
OUT_DIR.mkdir(exist_ok=True)
DB_PATH = OUT_DIR / 'fragment_index.duckdb'
CHUNK_DIR = OUT_DIR / 'chunks'
CHUNK_DIR.mkdir(exist_ok=True)

N_THREADS = 16
REPORT_EVERY = 500  # fragments

def connect():
    db = lancedb.connect(
        's3://esm-protein-atlas/v1/folds',
        storage_options={'aws_region': 'us-west-2', 'skip_signature': 'true'}
    )
    return db.open_table('folds_1B').to_lance()

def process_chunk(thread_id, frags):
    out_path = CHUNK_DIR / f'chunk_{thread_id:02d}.parquet'
    tmp_path = CHUNK_DIR / f'chunk_{thread_id:02d}.tmp.parquet'
    schema = pa.schema([('protein_hash', pa.string()), ('fragment_id', pa.int32()), ('frag_row', pa.int32())])

    done_frags = set()
    if out_path.exists():
        existing = pq.read_table(str(out_path), columns=['fragment_id'])
        done_frags = set(existing['fragment_id'].to_pylist())
        writer = pq.ParquetWriter(str(tmp_path), schema, compression='zstd')
    else:
        writer = None

    for i, frag in enumerate(frags):
        if frag.fragment_id in done_frags:
            continue
        try:
            tbl = frag.to_table(columns=['protein_hash'])
            n = len(tbl)
            frag_col = pa.array([frag.fragment_id] * n, type=pa.int32())
            frag_row = pa.array(range(n), type=pa.int32())
            batch = pa.table({'protein_hash': tbl['protein_hash'], 'fragment_id': frag_col, 'frag_row': frag_row})
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), schema, compression='zstd')
            writer.write_table(batch)
        except Exception as e:
            print(f'[thread {thread_id}] frag {frag.fragment_id} error: {e}')

        if (i + 1) % REPORT_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(frags) - i - 1) / rate / 3600
            print(f'[thread {thread_id:2d}] {i+1}/{len(frags)} frags | {rate:.1f} frags/s | ~{remaining:.1f}h left')

    if writer:
        writer.close()
    if tmp_path.exists():
        merged = pa.concat_tables([
            pq.read_table(str(out_path)),
            pq.read_table(str(tmp_path)),
        ])
        pq.write_table(merged, str(out_path), compression='zstd')
        tmp_path.unlink()
    return thread_id, len(frags)

t0 = time.time()

def main():
    global t0
    print('Connecting and listing fragments...')
    dataset = connect()
    frags = dataset.get_fragments()
    n_frags = len(frags)
    print(f'{n_frags} fragments total, splitting across {N_THREADS} threads')

    chunk_size = (n_frags + N_THREADS - 1) // N_THREADS
    splits = [frags[i*chunk_size:(i+1)*chunk_size] for i in range(N_THREADS)]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_THREADS) as exe:
        futures = {exe.submit(process_chunk, tid, splits[tid]): tid for tid in range(N_THREADS)}
        for fut in as_completed(futures):
            tid, count = fut.result()
            elapsed = time.time() - t0
            print(f'Thread {tid} finished {count} fragments in {elapsed/60:.1f}m')

    print(f'\nAll threads done in {(time.time()-t0)/3600:.2f}h. Building DuckDB index...')

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS fragment_index AS
        SELECT protein_hash, fragment_id
        FROM read_parquet('{CHUNK_DIR}/*.parquet')
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fragment_index (protein_hash)")
    count = con.execute("SELECT COUNT(*) FROM fragment_index").fetchone()[0]
    con.close()
    print(f'Done. {count:,} rows indexed at {DB_PATH}')
    print(f'Total time: {(time.time()-t0)/3600:.2f}h')

def build_index_only():
    """Merge chunk parquets into a single sorted parquet, partitioned by first hex char (16 files)."""
    import time
    t = time.time()
    MERGED_DIR = OUT_DIR / 'merged'
    MERGED_DIR.mkdir(exist_ok=True)
    print('Partitioning chunks into 16 sorted parquets by hash prefix...')
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")
    for prefix in '0123456789abcdef':
        out = MERGED_DIR / f'prefix_{prefix}.parquet'
        if out.exists():
            print(f'  prefix {prefix}: already done, skipping')
            continue
        con.execute(f"""
            COPY (
                SELECT protein_hash, fragment_id, frag_row
                FROM read_parquet('{CHUNK_DIR}/*.parquet')
                WHERE protein_hash LIKE '{prefix}%'
                ORDER BY protein_hash
            ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
        """)
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f'  prefix {prefix}: {count:,} rows -> {out.name}')
    con.close()
    print(f'Done in {(time.time()-t)/60:.1f}m. Query with lookup(hashes).')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--index-only':
        build_index_only()
    else:
        main()
