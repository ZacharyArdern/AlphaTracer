use recoded_sketch::{all_kmers, build_alphabet, col_strings, open_reader, total_rows};
use arrow_array::{LargeStringArray, RecordBatch, UInt32Array, FixedSizeListArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_N_HASH: usize = 100;
const DEFAULT_K: usize = 9;
const DEFAULT_MAX_FREQ_FRAC: f64 = 0.001;
const DEFAULT_SCHEME: &str = "murphy2000_5";

fn col_strings_any(batch: &RecordBatch, names: &[&str]) -> Vec<String> {
    for &name in names {
        if batch.column_by_name(name).is_some() {
            return col_strings(batch, name);
        }
    }
    panic!("none of {:?} found in batch", names);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: sketch <in1.parquet> [in2.parquet ...] <out.parquet> [k=9] [max_freq=0.001] [n_hash=100] [recoding_scheme=murphy2000_5]");
        eprintln!("  max_freq: fraction of DB (e.g. 0.001 = 0.1%) or absolute count if >= 1");
        std::process::exit(1);
    }

    let mut file_args:  Vec<usize> = Vec::new();
    let mut num_args:   Vec<usize> = Vec::new();
    let mut scheme_arg: Option<String> = None;
    for i in 1..args.len() {
        let a = &args[i];
        if a.ends_with(".parquet") || a.ends_with(".pq")       { file_args.push(i); }
        else if a.starts_with("murphy") || a.starts_with("dayhoff") { scheme_arg = Some(a.clone()); }
        else if a.parse::<f64>().is_ok()                       { num_args.push(i); }
        else { eprintln!("Unexpected argument: {a}"); std::process::exit(1); }
    }
    if file_args.len() < 2 {
        eprintln!("Need at least one input and one output parquet file");
        std::process::exit(1);
    }

    let out_path = args[*file_args.last().unwrap()].clone();
    let in_paths: Vec<String> = file_args[..file_args.len()-1].iter().map(|&i| args[i].clone()).collect();
    let k            = num_args.get(0).and_then(|&i| args[i].parse::<f64>().ok()).map(|v| v as usize).unwrap_or(DEFAULT_K);
    let max_freq_arg = num_args.get(1).and_then(|&i| args[i].parse::<f64>().ok()).unwrap_or(DEFAULT_MAX_FREQ_FRAC);
    let n_hash       = num_args.get(2).and_then(|&i| args[i].parse::<usize>().ok()).unwrap_or(DEFAULT_N_HASH);
    let scheme       = scheme_arg.as_deref().unwrap_or(DEFAULT_SCHEME);

    let (dayhoff_arr, n_letters) = build_alphabet(scheme);
    let base      = n_letters as u32;
    let flat_size = n_letters.pow(k as u32);
    let dayhoff   = Arc::new(dayhoff_arr);
    let t0        = Instant::now();
    let total     = total_rows(&in_paths);

    let max_freq: usize = if max_freq_arg < 1.0 {
        ((total as f64 * max_freq_arg) as usize).max(1)
    } else {
        max_freq_arg as usize
    };

    eprintln!("Inputs: {} file(s), {total} rows total", in_paths.len());
    eprintln!("Output: {out_path}  k={k}  n_hash={n_hash}  scheme={scheme}");
    eprintln!("max_freq: {max_freq} ({:.3}% of DB)", max_freq as f64 / total as f64 * 100.0);

    // ── Pass 1: flat k-mer frequency count ───────────────────────────────────
    // L3-cache-aware: cap fold threads so per-thread flat Vec fits in ~8 MB.
    const L3_BUDGET: usize = 8 * 1024 * 1024;
    let n_fold     = ((L3_BUDGET / (flat_size * 4)).max(1)).min(rayon::current_num_threads());
    let chunk_size = (100_000 / n_fold).max(1);

    eprintln!("Pass 1: k-mer frequencies (fold_threads={n_fold}) ...");
    let mut freq = vec![0u32; flat_size];
    let mut done = 0usize;

    for in_path in &in_paths {
        for batch in open_reader(in_path).build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_strings(&batch, "sequence");
            let dh    = Arc::clone(&dayhoff);
            let batch_freq: Vec<u32> = seqs
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut m = vec![0u32; flat_size];
                    for s in chunk {
                        for h in all_kmers(s.as_bytes(), &dh, k, base) {
                            m[h as usize] = m[h as usize].saturating_add(1);
                        }
                    }
                    m
                })
                .reduce(|| vec![0u32; flat_size], |mut a, b| {
                    for i in 0..flat_size { a[i] = a[i].saturating_add(b[i]); }
                    a
                });
            for i in 0..flat_size { freq[i] = freq[i].saturating_add(batch_freq[i]); }
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();

    let blocked = freq.iter().filter(|&&c| c as usize > max_freq).count();
    let nonzero = freq.iter().filter(|&&c| c > 0).count();
    eprintln!("  {blocked}/{nonzero} k-mers blocked (freq>{max_freq})  ({:.1}s)", t0.elapsed().as_secs_f64());

    let blocklist: Arc<Vec<bool>> = Arc::new(freq.iter().map(|&c| c as usize > max_freq).collect());
    drop(freq);

    // ── Pass 2: build filtered sketches ──────────────────────────────────────
    eprintln!("Pass 2: building sketches ...");

    let item_field = Arc::new(Field::new("item", DataType::UInt32, false));
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("AFDB_ID", DataType::LargeUtf8, true),
        Field::new("sketch",  DataType::FixedSizeList(item_field.clone(), n_hash as i32), false),
    ]));
    let props    = WriterProperties::builder().set_compression(Compression::ZSTD(Default::default())).build();
    let out_file = std::fs::File::create(&out_path).expect("create output");
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props)).unwrap();

    done = 0;
    let t1 = Instant::now();

    for in_path in &in_paths {
        for batch in open_reader(in_path).build().unwrap() {
            let batch    = batch.unwrap();
            let n        = batch.num_rows();
            let seqs     = col_strings(&batch, "sequence");
            let afdb_ids = col_strings_any(&batch, &["AFDB_ID", "rep_AFDB_ID"]);
            let dh = Arc::clone(&dayhoff);
            let bl = Arc::clone(&blocklist);

            let flat: Vec<u32> = seqs.par_iter().flat_map(|s| {
                let mut sk: Vec<u32> = all_kmers(s.as_bytes(), &dh, k, base)
                    .into_iter().filter(|&h| !bl[h as usize]).take(n_hash).collect();
                sk.resize(n_hash, u32::MAX);
                sk
            }).collect();

            let values     = Arc::new(UInt32Array::from(flat));
            let sketch_col = Arc::new(FixedSizeListArray::try_new(item_field.clone(), n_hash as i32, values, None).unwrap());
            let id_col     = Arc::new(LargeStringArray::from(afdb_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>()));

            writer.write(&RecordBatch::try_new(out_schema.clone(), vec![id_col, sketch_col]).unwrap()).unwrap();

            done += n;
            eprint!("  {done:>12}/{total}  ({:.0} seq/s)\r", done as f64 / t1.elapsed().as_secs_f64());
        }
    }
    eprintln!();

    writer.close().unwrap();
    eprintln!("\nDone. {done} sequences in {:.1}s total", t0.elapsed().as_secs_f64());
    eprintln!("Output: {:.2} GB", std::fs::metadata(&out_path).unwrap().len() as f64 / 1e9);
}
