/// Build a .sidx v2 directly from raw sequence parquets.
///
/// Three optimisations vs v1:
///   1. Flat pos_arr (Vec<u32> indexed by hash) instead of FxHashMap — O(1) direct lookup,
///      same cache footprint as the Pass 1 freq array, no hashing overhead.
///   2. Configurable N_HASH (default 64 vs 100) — postings shrink proportionally.
///   3. Sequences sorted by minimum sketch hash before indexing + VarInt delta-coded
///      posting lists — similar sequences cluster → small deltas → better compression.
///
/// Output format (.sidx v2):
///   [10] magic "SKETCHIDX2"
///   [8]  n_seqs        u64
///   [8]  n_hashes      u64
///   [8]  n_encoded     u64   (byte length of VarInt postings block)
///   [4]  max_freq      u32
///   [4]  n_hash        u32   (actual N_HASH used)
///   [n_hashes*4]       hash_keys  u32[]
///   [(n_hashes+1)*8]   offsets    u64[]  (byte offsets into encoded block)
///   [n_encoded bytes]  VarInt delta-coded posting lists
///   [n_seqs*4]         id_lengths u32[]
///   [sum(id_lengths)]  id_bytes   UTF-8
///
/// Usage: index-seqs <seq.parquet> [seq2.parquet ...] <out.sidx> [k=11] [max_freq=0.001] [n_hash=64]

use arrow_array::{Array, StringArray, LargeStringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_K: usize = 11;
const DEFAULT_N_HASH: usize = 64;
const CHUNK: usize = 100_000;
const L3_BUDGET: usize = 8 * 1024 * 1024;

fn build_dayhoff() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
    for &c in b"AaGgSsTtPp" { t[c as usize] = 1; }
    for &c in b"FfYyWw"     { t[c as usize] = 2; }
    for &c in b"EeDdNnQq"   { t[c as usize] = 3; }
    for &c in b"KkRrHh"     { t[c as usize] = 4; }
    t
}

fn open_reader(path: &str) -> ParquetRecordBatchReaderBuilder<std::fs::File> {
    let file = std::fs::File::open(path).expect("open parquet");
    ParquetRecordBatchReaderBuilder::try_new(file).unwrap().with_batch_size(CHUNK)
}

fn total_rows(paths: &[String]) -> usize {
    paths.iter().map(|p| open_reader(p).metadata().file_metadata().num_rows() as usize).sum()
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

fn col_strings_any(batch: &arrow_array::RecordBatch, names: &[&str]) -> Vec<String> {
    for &name in names {
        if batch.column_by_name(name).is_some() {
            return col_strings(batch, name);
        }
    }
    panic!("none of {:?} found in batch", names);
}

fn all_kmers(seq: &[u8], dayhoff: &[u8; 256], k: usize, pow_k: u32) -> Vec<u32> {
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
    out.sort_unstable();
    out.dedup();
    out
}

/// N_HASH smallest k-mers present in pos_arr (u32::MAX = not kept).
fn sketch_pos(seq: &[u8], dayhoff: &[u8; 256], k: usize, pow_k: u32,
              pos_arr: &[u32], n_hash: usize) -> Vec<u32> {
    let kmers = all_kmers(seq, dayhoff, k, pow_k);
    let mut out = Vec::with_capacity(n_hash);
    for &h in &kmers {
        if out.len() == n_hash { break; }
        if pos_arr[h as usize] != u32::MAX { out.push(h); }
    }
    out
}

fn write_varint(buf: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 { buf.push((v as u8) | 0x80); v >>= 7; }
    buf.push(v as u8);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: index-seqs <seq.parquet> [seq2.parquet ...] <out.sidx> [k=11] [max_freq=0.001] [n_hash=64]");
        std::process::exit(1);
    }

    let mut pq_args:  Vec<usize> = Vec::new();
    let mut sidx_arg: Option<usize> = None;
    let mut num_args: Vec<usize> = Vec::new();
    for i in 1..args.len() {
        let a = &args[i];
        if a.ends_with(".sidx")                              { sidx_arg = Some(i); }
        else if a.ends_with(".parquet") || a.ends_with(".pq") { pq_args.push(i); }
        else if a.parse::<f64>().is_ok()                     { num_args.push(i); }
        else { eprintln!("Unexpected argument: {a}"); std::process::exit(1); }
    }
    let out_path = match sidx_arg {
        Some(i) => args[i].clone(),
        None => { eprintln!("No .sidx output path"); std::process::exit(1); }
    };
    if pq_args.is_empty() { eprintln!("No input parquets"); std::process::exit(1); }

    let seq_pqs:     Vec<String> = pq_args.iter().map(|&i| args[i].clone()).collect();
    let k        = num_args.get(0).and_then(|&i| args[i].parse::<f64>().ok()).map(|v| v as usize).unwrap_or(DEFAULT_K);
    let mf_arg   = num_args.get(1).and_then(|&i| args[i].parse::<f64>().ok()).unwrap_or(0.001);
    let n_hash   = num_args.get(2).and_then(|&i| args[i].parse::<usize>().ok()).unwrap_or(DEFAULT_N_HASH);

    let flat_size = 5usize.pow(k as u32);
    let pow_k     = 5u32.pow((k - 1) as u32);
    let dayhoff   = Arc::new(build_dayhoff());
    let t0        = Instant::now();

    let total    = total_rows(&seq_pqs);
    let max_freq: usize = if mf_arg < 1.0 { ((total as f64 * mf_arg) as usize).max(1) } else { mf_arg as usize };
    eprintln!("{} file(s), {total} seqs  k={k}  n_hash={n_hash}  max_freq={max_freq} ({:.3}%)",
              seq_pqs.len(), max_freq as f64 / total as f64 * 100.0);

    // ── Pass 1: flat k-mer frequency count ───────────────────────────────────
    let n_fold = (L3_BUDGET / (flat_size * 4)).max(1).min(rayon::current_num_threads());
    let chunk_size = (CHUNK / n_fold).max(1);
    eprintln!("Pass 1: k-mer frequencies ({:.0} MB flat array, fold_threads={n_fold}) ...",
              flat_size as f64 * 4.0 / 1e6);
    let mut freq = vec![0u32; flat_size];
    let mut done = 0usize;
    for pq in &seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_strings(&batch, "sequence");
            let dh    = Arc::clone(&dayhoff);
            let bf: Vec<u32> = seqs.par_chunks(chunk_size).map(|chunk| {
                let mut m = vec![0u32; flat_size];
                for s in chunk {
                    for h in all_kmers(s.as_bytes(), &dh, k, pow_k) {
                        m[h as usize] = m[h as usize].saturating_add(1);
                    }
                }
                m
            }).reduce(|| vec![0u32; flat_size], |mut a, b| {
                for i in 0..flat_size { a[i] = a[i].saturating_add(b[i]); }
                a
            });
            for i in 0..flat_size { freq[i] = freq[i].saturating_add(bf[i]); }
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();
    eprintln!("  Pass 1 done  ({:.1}s)", t0.elapsed().as_secs_f64());

    // Build flat pos_arr: pos_arr[h] = position in hash_keys, or u32::MAX if filtered.
    // Direct O(1) array lookup replaces FxHashMap for all subsequent passes.
    let mut hash_keys: Vec<u32> = freq.iter().enumerate()
        .filter(|(_, &c)| c as usize > 0 && c as usize <= max_freq)
        .map(|(h, _)| h as u32).collect();
    drop(freq);
    hash_keys.sort_unstable();

    let mut pos_arr = vec![u32::MAX; flat_size];
    for (i, &h) in hash_keys.iter().enumerate() { pos_arr[h as usize] = i as u32; }
    let pos_arr = Arc::new(pos_arr);

    eprintln!("  {} candidate hashes  ({:.1}s)", hash_keys.len(), t0.elapsed().as_secs_f64());

    // ── Pass 1b (CSR, 2-pass): avoids Vec<Vec> capacity overhead (2×) ────────────
    // Pass 1 of 2: count sketch appearances per candidate hash → exact allocation.
    let n_cand_hashes = hash_keys.len();
    let mut sketch_count: Vec<u32> = vec![0u32; n_cand_hashes];
    let mut db_ids: Vec<String> = Vec::with_capacity(total);
    done = 0;

    eprintln!("Pass 1b (1/2): counting sketch appearances ...");
    for pq in &seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            db_ids.extend(col_strings_any(&batch, &["AFDB_ID", "rep_AFDB_ID"]));
            let seqs = col_strings(&batch, "sequence");
            let pa = Arc::clone(&pos_arr);
            let dh = Arc::clone(&dayhoff);
            let batch_sketches: Vec<Vec<u32>> = seqs.par_iter()
                .map(|s| sketch_pos(s.as_bytes(), &dh, k, pow_k, &pa, n_hash))
                .collect();
            for sk in &batch_sketches {
                for &h in sk {
                    sketch_count[pos_arr[h as usize] as usize] += 1;
                }
            }
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();
    let n_seqs = db_ids.len();

    // Prune hash_keys to those with at least one sketch appearance, rebuild pos_arr.
    let old_pos_arr = Arc::try_unwrap(pos_arr).unwrap();
    drop(old_pos_arr);
    let (hash_keys, sketch_count): (Vec<u32>, Vec<u32>) = hash_keys.into_iter()
        .zip(sketch_count.into_iter())
        .filter(|(_, c)| *c > 0)
        .unzip();
    let n_hashes = hash_keys.len();
    eprintln!("  {n_hashes} hashes in sketches  n_seqs={n_seqs}  ({:.1}s)", t0.elapsed().as_secs_f64());

    let mut pos_arr2 = vec![u32::MAX; flat_size];
    for (i, &h) in hash_keys.iter().enumerate() { pos_arr2[h as usize] = i as u32; }
    let pos_arr2 = Arc::new(pos_arr2);

    // Compute postings-array offsets via prefix sum (u64 — n_postings can exceed u32::MAX).
    let n_postings: usize = sketch_count.iter().map(|&c| c as usize).sum();
    let mut post_offsets: Vec<u64> = Vec::with_capacity(n_hashes + 1);
    let mut cumsum = 0u64;
    for &c in &sketch_count {
        post_offsets.push(cumsum);
        cumsum += c as u64;
    }
    post_offsets.push(cumsum);
    drop(sketch_count);

    // Single flat postings array — no capacity overhead, one contiguous allocation.
    let mut postings: Vec<u32> = vec![0u32; n_postings];
    let mut cursors: Vec<u64> = post_offsets[..n_hashes].to_vec();

    // Pass 2 of 2: re-read parquet, fill flat postings.
    eprintln!("Pass 1b (2/2): filling flat postings ({:.2} GB) ...", n_postings as f64 * 4.0 / 1e9);
    done = 0;
    let mut seq_id = 0u32;

    for pq in &seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            let seqs = col_strings(&batch, "sequence");
            let pa = Arc::clone(&pos_arr2);
            let dh = Arc::clone(&dayhoff);
            let batch_sketches: Vec<Vec<u32>> = seqs.par_iter()
                .map(|s| sketch_pos(s.as_bytes(), &dh, k, pow_k, &pa, n_hash))
                .collect();
            for sk in &batch_sketches {
                for &h in sk {
                    let pos = pos_arr2[h as usize] as usize;
                    postings[cursors[pos] as usize] = seq_id;
                    cursors[pos] += 1;
                }
                seq_id += 1;
            }
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();
    drop(cursors);
    eprintln!("  {n_postings} postings  ({:.1}s)", t0.elapsed().as_secs_f64());

    let db_ids_sorted = db_ids;

    // Sort each posting list subrange (postings are in parquet order).
    eprintln!("Sorting posting lists ...");
    for i in 0..n_hashes {
        let s = post_offsets[i] as usize;
        let e = post_offsets[i + 1] as usize;
        if s < e { postings[s..e].sort_unstable(); }
    }
    eprintln!("  sorted  ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── VarInt delta-encode posting lists + build byte offsets ───────────────
    eprintln!("Delta-encoding posting lists ...");
    let mut encoded: Vec<u8> = Vec::with_capacity(n_postings * 2);
    let mut offsets: Vec<u64> = Vec::with_capacity(n_hashes + 1);
    for i in 0..n_hashes {
        offsets.push(encoded.len() as u64);
        let s = post_offsets[i] as usize;
        let e = post_offsets[i + 1] as usize;
        if s == e { continue; }
        let mut prev = 0u32;
        for &seq_id in &postings[s..e] {
            write_varint(&mut encoded, seq_id - prev);
            prev = seq_id;
        }
    }
    offsets.push(encoded.len() as u64);
    let n_encoded = encoded.len();
    drop(postings); drop(post_offsets);
    eprintln!("  encoded: {:.2} GB  (raw: {:.2} GB, ratio {:.2}x)  ({:.1}s)",
              n_encoded as f64 / 1e9,
              n_postings as f64 * 4.0 / 1e9,
              n_postings as f64 * 4.0 / n_encoded as f64,
              t0.elapsed().as_secs_f64());

    // ── Write .sidx v2 ────────────────────────────────────────────────────────
    eprintln!("Writing {out_path} ...");
    let f = std::fs::File::create(&out_path).expect("create index");
    let mut w = BufWriter::with_capacity(64 * 1024 * 1024, f);

    w.write_all(b"SKETCHIDX2").unwrap();
    w.write_all(&(n_seqs     as u64).to_le_bytes()).unwrap();
    w.write_all(&(n_hashes   as u64).to_le_bytes()).unwrap();
    w.write_all(&(n_encoded  as u64).to_le_bytes()).unwrap();
    w.write_all(&(max_freq   as u32).to_le_bytes()).unwrap();
    w.write_all(&(n_hash     as u32).to_le_bytes()).unwrap();
    for &h in &hash_keys { w.write_all(&h.to_le_bytes()).unwrap(); }
    for &o in &offsets   { w.write_all(&o.to_le_bytes()).unwrap(); }
    w.write_all(&encoded).unwrap();
    for id in &db_ids_sorted { w.write_all(&(id.len() as u32).to_le_bytes()).unwrap(); }
    for id in &db_ids_sorted { w.write_all(id.as_bytes()).unwrap(); }
    w.flush().unwrap();

    let sz = std::fs::metadata(&out_path).unwrap().len();
    eprintln!("Done. {:.1}s total  |  {:.3} GB", t0.elapsed().as_secs_f64(), sz as f64 / 1e9);
}
