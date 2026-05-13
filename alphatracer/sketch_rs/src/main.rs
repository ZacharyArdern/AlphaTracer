use arrow_array::{
    Array, RecordBatch, StringArray, LargeStringArray,
    FixedSizeListArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const N_HASH: usize = 100;
const CHUNK: usize = 100_000;
const DEFAULT_MAX_FREQ_FRAC: f64 = 0.001; // 0.1% of database
const DEFAULT_K: usize = 9;

// 5-group Dayhoff (Murphy et al. 2000):
//   0: LVIMC  (hydrophobic + cysteine)
//   1: AGSTP  (small/flexible)
//   2: FYW    (aromatic)
//   3: EDNQ   (polar/acidic)
//   4: KRH    (basic)
//   0xFF: unknown — k-mers containing this are skipped
fn build_dayhoff() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
    for &c in b"AaGgSsTtPp" { t[c as usize] = 1; }
    for &c in b"FfYyWw"     { t[c as usize] = 2; }
    for &c in b"EeDdNnQq"   { t[c as usize] = 3; }
    for &c in b"KkRrHh"     { t[c as usize] = 4; }
    t
}

// Base-5 rolling k-mer index. Returns indices in [0, 5^k).
// h_i     = d[i]*5^(k-1) + ... + d[i+k-1]
// h_{i+1} = (h_i - d[i]*5^(k-1)) * 5 + d[i+k]
// k-mers touching an unknown AA (0xFF) are skipped.
fn kmers_flat(seq: &[u8], dayhoff: &[u8; 256], k: usize, pow_k: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
    let mut seg_start = 0usize;
    for i in 0..=seq.len() {
        if i < seq.len() && dayhoff[seq[i] as usize] != 0xFF { continue; }
        let seg = &seq[seg_start..i];
        if seg.len() >= k {
            let mut h: u32 = 0;
            for j in 0..k { h = h * 5 + dayhoff[seg[j] as usize] as u32; }
            out.push(h);
            for s in 1..=seg.len() - k {
                h = h.wrapping_sub(pow_k * dayhoff[seg[s-1] as usize] as u32) * 5
                    + dayhoff[seg[s+k-1] as usize] as u32;
                out.push(h);
            }
        }
        seg_start = i + 1;
    }
    out
}

// Pass-2 sketch: segment-based rolling hash, filters against the flat blocklist.
fn sketch(seq: &[u8], dayhoff: &[u8; 256], k: usize, pow_k: u32, blocklist: &[bool]) -> [u32; N_HASH] {
    let mut hashes: Vec<u32> = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
    let mut seg_start = 0usize;
    for i in 0..=seq.len() {
        if i < seq.len() && dayhoff[seq[i] as usize] != 0xFF { continue; }
        let seg = &seq[seg_start..i];
        if seg.len() >= k {
            let mut h: u32 = 0;
            for j in 0..k { h = h * 5 + dayhoff[seg[j] as usize] as u32; }
            if !blocklist[h as usize] { hashes.push(h); }
            for s in 1..=seg.len() - k {
                h = h.wrapping_sub(pow_k * dayhoff[seg[s-1] as usize] as u32) * 5
                    + dayhoff[seg[s+k-1] as usize] as u32;
                if !blocklist[h as usize] { hashes.push(h); }
            }
        }
        seg_start = i + 1;
    }

    hashes.sort_unstable();
    hashes.dedup();
    let mut out = [u32::MAX; N_HASH];
    let m = hashes.len().min(N_HASH);
    out[..m].copy_from_slice(&hashes[..m]);
    out
}

fn col_as_strings(batch: &RecordBatch, name: &str) -> Vec<String> {
    let col = batch.column_by_name(name).expect(name);
    let n   = col.len();
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        (0..n).map(|i| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        (0..n).map(|i| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).collect()
    } else {
        vec![String::new(); n]
    }
}

// Try each name in order; returns the first that exists in the batch.
fn col_as_strings_any(batch: &RecordBatch, names: &[&str]) -> Vec<String> {
    for &name in names {
        if batch.column_by_name(name).is_some() {
            return col_as_strings(batch, name);
        }
    }
    panic!("none of {:?} found in batch", names);
}

fn open_reader(path: &str) -> (ParquetRecordBatchReaderBuilder<std::fs::File>, i64) {
    let file    = std::fs::File::open(path).expect("open input");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .with_batch_size(CHUNK);
    let total = builder.metadata().file_metadata().num_rows();
    (builder, total)
}

fn total_rows(paths: &[String]) -> i64 {
    paths.iter().map(|p| open_reader(p).1).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: sketch <in1.parquet> [in2.parquet ...] <out.parquet> [k=9] [max_freq=0.001]");
        eprintln!("  max_freq: fraction of DB (e.g. 0.001 = 0.1%) or absolute count if >= 1");
        std::process::exit(1);
    }

    let mut file_args: Vec<usize> = Vec::new();
    let mut num_args:  Vec<usize> = Vec::new();
    for i in 1..args.len() {
        let a = &args[i];
        if a.ends_with(".parquet") || a.ends_with(".pq") {
            file_args.push(i);
        } else if a.parse::<f64>().is_ok() {
            num_args.push(i);
        } else {
            eprintln!("Unexpected argument: {a}");
            std::process::exit(1);
        }
    }
    if file_args.len() < 2 {
        eprintln!("Need at least one input and one output parquet file");
        std::process::exit(1);
    }
    let out_path     = args[*file_args.last().unwrap()].clone();
    let in_paths: Vec<String> = file_args[..file_args.len()-1].iter().map(|&i| args[i].clone()).collect();
    let k            = num_args.get(0).and_then(|&i| args[i].parse::<f64>().ok()).map(|v| v as usize).unwrap_or(DEFAULT_K);
    let max_freq_arg = num_args.get(1).and_then(|&i| args[i].parse::<f64>().ok()).unwrap_or(DEFAULT_MAX_FREQ_FRAC);

    let flat_size = 5usize.pow(k as u32);
    let pow_k     = 5u32.pow((k - 1) as u32);
    let dayhoff   = Arc::new(build_dayhoff());
    let t0        = Instant::now();
    let total     = total_rows(&in_paths);

    // Resolve max_freq: fraction < 1.0 → relative to DB size; else absolute.
    let max_freq: usize = if max_freq_arg < 1.0 {
        ((total as f64 * max_freq_arg) as usize).max(1)
    } else {
        max_freq_arg as usize
    };
    let max_freq_pct = max_freq as f64 / total as f64 * 100.0;

    eprintln!("Inputs: {} file(s), {total} rows total", in_paths.len());
    eprintln!("Output: {out_path}  k={k}, n_hashes={N_HASH}");
    eprintln!("max_freq: {max_freq} ({max_freq_pct:.3}% of DB)");
    eprintln!("Flat freq table: {flat_size} entries ({:.0} KB)",
              flat_size as f64 * 4.0 / 1024.0);

    // ── Pass 1: flat-array frequency count ───────────────────────────────────
    // Number of fold threads is capped so all per-thread Vecs fit in ~8 MB of L3.
    // k=6 (61 KB) → 8 threads; k=9 (7.6 MB) → 1 thread (sequential fold, one Vec in L3).
    const L3_BUDGET: usize = 8 * 1024 * 1024;
    let n_fold = ((L3_BUDGET / (flat_size * 4)).max(1))
        .min(rayon::current_num_threads());
    let chunk_size = (CHUNK / n_fold).max(1);

    eprintln!("Pass 1: counting k-mer frequencies (k={k}, max_freq={max_freq}, fold_threads={n_fold}) ...");
    let mut freq = vec![0u32; flat_size];
    let mut done = 0usize;

    for in_path in &in_paths {
        let (builder, _) = open_reader(in_path);
        for batch in builder.build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_as_strings(&batch, "sequence");
            let dh    = Arc::clone(&dayhoff);

            // par_chunks → each chunk builds one flat Vec → reduce with element-wise add.
            let batch_freq: Vec<u32> = seqs
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut m = vec![0u32; flat_size];
                    for s in chunk {
                        for h in kmers_flat(s.as_bytes(), &dh, k, pow_k) {
                            m[h as usize] = m[h as usize].saturating_add(1);
                        }
                    }
                    m
                })
                .reduce(
                    || vec![0u32; flat_size],
                    |mut a, b| {
                        for i in 0..flat_size { a[i] = a[i].saturating_add(b[i]); }
                        a
                    },
                );

            for i in 0..flat_size { freq[i] = freq[i].saturating_add(batch_freq[i]); }
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();

    let blocked = freq.iter().filter(|&&c| c as usize > max_freq).count();
    let nonzero = freq.iter().filter(|&&c| c > 0).count();
    eprintln!("  {blocked}/{nonzero} unique k-mers blocked (freq>{max_freq})  ({:.1}s)",
              t0.elapsed().as_secs_f64());

    // Flat blocklist: direct array lookup, no hashing.
    let blocklist: Arc<Vec<bool>> = Arc::new(freq.iter().map(|&c| c as usize > max_freq).collect());
    drop(freq);

    // ── Pass 2: build filtered sketches ──────────────────────────────────────
    eprintln!("Pass 2: building sketches ...");

    let item_field = Arc::new(Field::new("item", DataType::UInt32, false));
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("AFDB_ID", DataType::LargeUtf8, true),
        Field::new("sketch",  DataType::FixedSizeList(item_field.clone(), N_HASH as i32), false),
    ]));

    let props    = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let out_file = std::fs::File::create(&out_path).expect("create output");
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props)).unwrap();

    done = 0;
    let t1 = Instant::now();

    for in_path in &in_paths {
        let (builder2, _) = open_reader(in_path);
        for batch in builder2.build().unwrap() {
            let batch    = batch.unwrap();
            let n        = batch.num_rows();
            let seqs     = col_as_strings(&batch, "sequence");
            let afdb_ids = col_as_strings_any(&batch, &["AFDB_ID", "rep_AFDB_ID"]);

            let dh = Arc::clone(&dayhoff);
            let bl = Arc::clone(&blocklist);
            let sketches: Vec<[u32; N_HASH]> = seqs
                .par_iter()
                .map(|s| sketch(s.as_bytes(), &dh, k, pow_k, &bl))
                .collect();

            let flat: Vec<u32> = sketches.iter().flat_map(|s| s.iter().copied()).collect();
            let values     = Arc::new(UInt32Array::from(flat));
            let sketch_col = Arc::new(
                FixedSizeListArray::try_new(item_field.clone(), N_HASH as i32, values, None).unwrap()
            );
            let id_col = Arc::new(LargeStringArray::from(
                afdb_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            ));

            writer.write(&RecordBatch::try_new(
                out_schema.clone(), vec![id_col, sketch_col]
            ).unwrap()).unwrap();

            done += n;
            let secs = t1.elapsed().as_secs_f64();
            eprint!("  {done:>12}/{total}  ({:.0} seq/s)\r", done as f64 / secs);
        }
    }
    eprintln!();

    writer.close().unwrap();
    let secs = t0.elapsed().as_secs_f64();
    eprintln!("\nDone. {done} sequences in {secs:.1}s total");
    let sz = std::fs::metadata(&out_path).unwrap().len();
    eprintln!("Output: {:.2} GB", sz as f64 / 1e9);
}
