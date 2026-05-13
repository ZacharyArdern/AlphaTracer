/// Build a compact binary CSR inverted index from a sketch parquet.
/// Uses a two-pass approach to avoid HashMap<u32,Vec<u32>> growth/cache-miss slowdown:
///   Pass 1 — count occurrences per hash (small HashMap<u32,u32>, stays cache-friendly)
///   Pass 2 — fill pre-allocated flat postings array (sequential writes, no allocation)
///
/// Output format (.sidx):
///   [9]  magic "SKETCHIDX"
///   [8]  n_seqs        u64
///   [8]  n_hashes      u64
///   [8]  n_postings    u64
///   [4]  max_freq      u32
///   [4]  n_hash_const  u32
///   [n_hashes * 4]     hash_keys   u32[] sorted
///   [(n_hashes+1) * 8] offsets     u64[]
///   [n_postings * 4]   postings    u32[]
///   [n_seqs * 4]       id_lengths  u32[]
///   [sum(id_lengths)]  id_bytes    UTF-8

use arrow_array::{Array, StringArray, LargeStringArray, FixedSizeListArray, UInt32Array};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rustc_hash::FxHashMap;
use std::io::{BufWriter, Write};
use std::time::Instant;

const CHUNK: usize = 100_000;
const DEFAULT_MAX_FREQ: usize = 500;

fn open_reader(path: &str) -> ParquetRecordBatchReaderBuilder<std::fs::File> {
    let file = std::fs::File::open(path).expect("open parquet");
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .with_batch_size(CHUNK)
}

fn col_strings(batch: &arrow_array::RecordBatch, name: &str) -> Vec<String> {
    let col = batch.column_by_name(name).expect(name);
    let n = col.len();
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        (0..n).map(|i| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        (0..n).map(|i| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).collect()
    } else {
        vec![String::new(); n]
    }
}

/// Iterate hash values in a sketch column batch, calling f(seq_idx, hash) for each.
fn iter_hashes(batch: &arrow_array::RecordBatch, base_idx: u32, mut f: impl FnMut(u32, u32)) {
    let sk_col = batch.column_by_name("sketch").unwrap();
    let n = batch.num_rows();
    if let Some(fsl) = sk_col.as_any().downcast_ref::<FixedSizeListArray>() {
        let values = fsl.values().as_any().downcast_ref::<UInt32Array>().unwrap();
        let stride = fsl.value_length() as usize;
        for i in 0..n {
            let start = i * stride;
            for j in 0..stride {
                let h = values.value(start + j);
                if h != u32::MAX { f(base_idx + i as u32, h); }
            }
        }
    } else {
        let list_arr = sk_col.as_list::<i32>();
        for i in 0..n {
            let values   = list_arr.value(i);
            let uint_arr = values.as_primitive::<UInt32Type>();
            for j in 0..uint_arr.len() {
                let h = uint_arr.value(j);
                if h != u32::MAX { f(base_idx + i as u32, h); }
            }
        }
    }
}

fn total_rows(paths: &[String]) -> usize {
    paths.iter().map(|p| open_reader(p).metadata().file_metadata().num_rows() as usize).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Usage: build-index <sketch1.pq> [sketch2.pq ...] <output.sidx> [max_freq]
    // max_freq: fraction < 1.0 (e.g. 0.001 = 0.1% of DB) or absolute count if >= 1.
    if args.len() < 3 {
        eprintln!("Usage: build-index <sketch1.pq> [sketch2.pq ...] <output.sidx> [max_freq=0.001]");
        std::process::exit(1);
    }

    let mut pq_args: Vec<usize> = Vec::new();
    let mut sidx_arg: Option<usize> = None;
    let mut num_arg: Option<usize> = None;
    for i in 1..args.len() {
        let a = &args[i];
        if a.ends_with(".sidx") {
            sidx_arg = Some(i);
        } else if a.ends_with(".parquet") || a.ends_with(".pq") {
            pq_args.push(i);
        } else if a.parse::<f64>().is_ok() {
            num_arg = Some(i);
        } else {
            eprintln!("Unexpected argument: {a}");
            std::process::exit(1);
        }
    }
    let out_path = match sidx_arg {
        Some(i) => args[i].clone(),
        None    => { eprintln!("No .sidx output path specified"); std::process::exit(1); }
    };
    if pq_args.is_empty() {
        eprintln!("No input parquet files specified"); std::process::exit(1);
    }
    let sketch_pqs: Vec<String> = pq_args.iter().map(|&i| args[i].clone()).collect();
    let max_freq_arg: f64 = num_arg.and_then(|i| args[i].parse().ok()).unwrap_or(0.001);

    let t0 = Instant::now();
    let total = total_rows(&sketch_pqs);

    let max_freq: usize = if max_freq_arg < 1.0 {
        ((total as f64 * max_freq_arg) as usize).max(1)
    } else {
        max_freq_arg as usize
    };
    let max_freq_pct = max_freq as f64 / total as f64 * 100.0;
    eprintln!("{} input file(s), {total} seqs total  max_freq={max_freq} ({max_freq_pct:.3}% of DB)",
              sketch_pqs.len());

    // ── Pass 1: count occurrences per hash across all inputs ──────────────────
    eprintln!("Pass 1: counting hash frequencies ...");
    let mut counts: FxHashMap<u32, u32> = FxHashMap::with_capacity_and_hasher(
        4_000_000, Default::default()
    );
    let mut done = 0usize;
    for pq in &sketch_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            iter_hashes(&batch, 0, |_seq, h| {
                *counts.entry(h).or_insert(0) += 1;
            });
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}  ({} unique hashes)\r", counts.len());
        }
    }
    eprintln!();
    eprintln!("  {} unique hashes found  ({:.1}s)", counts.len(), t0.elapsed().as_secs_f64());

    let mut hash_keys: Vec<u32> = counts.iter()
        .filter(|(_, &c)| c as usize <= max_freq)
        .map(|(&h, _)| h)
        .collect();
    drop(counts);
    hash_keys.sort_unstable();

    let n_hashes = hash_keys.len();
    eprintln!("  {n_hashes} hashes kept after cap  ({:.1}s)", t0.elapsed().as_secs_f64());

    let hash_pos: FxHashMap<u32, u32> = hash_keys.iter().enumerate()
        .map(|(i, &h)| (h, i as u32))
        .collect();

    // Pass 1b: recount kept hashes for prefix sum
    eprintln!("Pass 1b: computing offsets ...");
    let mut hash_count = vec![0u32; n_hashes];
    done = 0;
    for pq in &sketch_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            iter_hashes(&batch, 0, |_seq, h| {
                if let Some(&pos) = hash_pos.get(&h) {
                    hash_count[pos as usize] += 1;
                }
            });
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();

    let mut offsets = vec![0u64; n_hashes + 1];
    for i in 0..n_hashes {
        offsets[i + 1] = offsets[i] + hash_count[i] as u64;
    }
    let n_postings = offsets[n_hashes] as usize;
    eprintln!("  {n_postings} total postings  ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Pass 2: fill postings ─────────────────────────────────────────────────
    eprintln!("Pass 2: filling postings array ...");
    let mut postings = vec![0u32; n_postings];
    let mut cursor   = offsets[..n_hashes].to_vec();
    let mut db_ids: Vec<String> = Vec::with_capacity(total);
    done = 0;

    for pq in &sketch_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch    = batch.unwrap();
            let base_idx = db_ids.len() as u32;
            db_ids.extend(col_strings(&batch, "AFDB_ID"));

            iter_hashes(&batch, base_idx, |seq_idx, h| {
                if let Some(&pos) = hash_pos.get(&h) {
                    let slot = cursor[pos as usize] as usize;
                    postings[slot] = seq_idx;
                    cursor[pos as usize] += 1;
                }
            });

            done += batch.num_rows();
            let secs = t0.elapsed().as_secs_f64();
            eprint!("  {done:>12}/{total}  ({:.0} seq/s)\r", done as f64 / secs);
        }
    }
    eprintln!();
    eprintln!("Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // ── Write binary index ────────────────────────────────────────────────────
    let n_seqs   = db_ids.len();
    let n_hash_c = (postings.len() / n_seqs.max(1)) as u32;
    eprintln!("Writing {out_path} ...");
    let f  = std::fs::File::create(&out_path).expect("create index");
    let mut w = BufWriter::with_capacity(64 * 1024 * 1024, f);

    w.write_all(b"SKETCHIDX").unwrap();
    w.write_all(&(n_seqs     as u64).to_le_bytes()).unwrap();
    w.write_all(&(n_hashes   as u64).to_le_bytes()).unwrap();
    w.write_all(&(n_postings as u64).to_le_bytes()).unwrap();
    w.write_all(&(max_freq   as u32).to_le_bytes()).unwrap();
    w.write_all(&n_hash_c.to_le_bytes()).unwrap();

    for &h in &hash_keys { w.write_all(&h.to_le_bytes()).unwrap(); }
    for &o in &offsets   { w.write_all(&o.to_le_bytes()).unwrap(); }
    for &p in &postings  { w.write_all(&p.to_le_bytes()).unwrap(); }
    for id in &db_ids    { w.write_all(&(id.len() as u32).to_le_bytes()).unwrap(); }
    for id in &db_ids    { w.write_all(id.as_bytes()).unwrap(); }

    w.flush().unwrap();
    let sz = std::fs::metadata(&out_path).unwrap().len();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("Done. {elapsed:.1}s total  |  {:.2} GB", sz as f64 / 1e9);
}
