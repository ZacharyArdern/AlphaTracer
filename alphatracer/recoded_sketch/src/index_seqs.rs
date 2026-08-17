/// Build a .sidx v4 directly from raw sequence parquets.
///
/// Output format (.sidx v4):
///   [10] magic "SKETCHIDX4"
///   [8]  n_seqs        u64
///   [8]  n_hashes      u64
///   [8]  n_encoded     u64   (byte length of encoded postings block)
///   [4]  max_freq      u32
///   [4]  n_hash        u32
///   [4]  k             u32
///   [32] recoding_scheme  null-padded UTF-8
///   [n_hashes*4]       hash_keys  u32[]
///   [(n_hashes+1)*8]   offsets    u64[]  (byte offsets into encoded block)
///   [n_encoded bytes]  VarInt delta-coded posting lists
///
/// Usage: index-seqs <seq.parquet> [seq2.parquet ...] <out.sidx> [k=11] [max_freq=0.001] [n_hash=64] [recoding_scheme=murphy2000_5]

use recoded_sketch::{all_kmers, build_alphabet, col_strings, open_reader, sketch_pos, total_rows, write_varint};
use rayon::prelude::*;
use std::io::{BufWriter, Write, Read};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_K: usize = 11;
const DEFAULT_N_HASH: usize = 64;
const DEFAULT_SCHEME: &str = "murphy2000_5";
const SHARD_TARGET_BYTES: usize = 8_000_000_000;

fn encode_shard(postings: &[u32], post_offsets: &[u64]) -> (Vec<u64>, Vec<u8>) {
    let n = post_offsets.len() - 1;
    let mut offsets: Vec<u64> = Vec::with_capacity(n + 1);
    let mut encoded: Vec<u8>  = Vec::new();
    for i in 0..n {
        offsets.push(encoded.len() as u64);
        let s = post_offsets[i] as usize;
        let e = post_offsets[i + 1] as usize;
        let mut prev = 0u32;
        for &id in &postings[s..e] {
            write_varint(id - prev, &mut encoded);
            prev = id;
        }
    }
    offsets.push(encoded.len() as u64);
    (offsets, encoded)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: index-seqs <seq.parquet> [seq2.parquet ...] <out.sidx> [k=11] [max_freq=0.001] [n_hash=64] [recoding_scheme=murphy2000_5]");
        eprintln!("  recoding_scheme: murphy2000_4 | murphy2000_5 | murphy2000_8 | dayhoff1978_6");
        std::process::exit(1);
    }

    let mut pq_args:     Vec<usize> = Vec::new();
    let mut sidx_arg:    Option<usize> = None;
    let mut num_args:    Vec<usize> = Vec::new();
    let mut scheme_arg:  Option<String> = None;
    for i in 1..args.len() {
        let a = &args[i];
        if a.ends_with(".sidx")                                { sidx_arg = Some(i); }
        else if a.ends_with(".parquet") || a.ends_with(".pq") { pq_args.push(i); }
        else if a.parse::<f64>().is_ok()                       { num_args.push(i); }
        else if a.starts_with("murphy") || a.starts_with("dayhoff") { scheme_arg = Some(a.clone()); }
        else { eprintln!("Unexpected argument: {a}"); std::process::exit(1); }
    }
    let out_path = match sidx_arg {
        Some(i) => args[i].clone(),
        None => { eprintln!("No .sidx output path"); std::process::exit(1); }
    };
    if pq_args.is_empty() { eprintln!("No input parquets"); std::process::exit(1); }

    let seq_pqs  = pq_args.iter().map(|&i| args[i].clone()).collect::<Vec<_>>();
    let k        = num_args.get(0).and_then(|&i| args[i].parse::<f64>().ok()).map(|v| v as usize).unwrap_or(DEFAULT_K);
    let mf_arg   = num_args.get(1).and_then(|&i| args[i].parse::<f64>().ok()).unwrap_or(0.001);
    let n_hash   = num_args.get(2).and_then(|&i| args[i].parse::<usize>().ok()).unwrap_or(DEFAULT_N_HASH);
    let scheme   = scheme_arg.as_deref().unwrap_or(DEFAULT_SCHEME);

    let (dayhoff_arr, n_letters) = build_alphabet(scheme);
    let base      = n_letters as u32;
    let flat_size = n_letters.pow(k as u32);
    let dayhoff   = Arc::new(dayhoff_arr);
    let t0        = Instant::now();

    let total    = total_rows(&seq_pqs);
    let max_freq = if mf_arg < 1.0 { ((total as f64 * mf_arg) as usize).max(1) } else { mf_arg as usize };
    eprintln!("{} file(s), {total} seqs  k={k}  scheme={scheme}  n_hash={n_hash}  max_freq={max_freq} ({:.3}%)",
              seq_pqs.len(), max_freq as f64 / total as f64 * 100.0);

    // ── Pass 1: flat k-mer frequency count ───────────────────────────────────
    let n_threads = rayon::current_num_threads();
    eprintln!("Pass 1: k-mer frequencies ({:.0} MB flat array, threads={n_threads}) ...",
              flat_size as f64 * 4.0 / 1e6);
    let freq_atomic: Arc<Vec<AtomicU32>> = Arc::new(
        (0..flat_size).map(|_| AtomicU32::new(0)).collect()
    );
    let mut done = 0usize;
    for pq in &seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_strings(&batch, "sequence");
            let dh    = Arc::clone(&dayhoff);
            let fa    = Arc::clone(&freq_atomic);
            seqs.par_iter().for_each(|s| {
                for h in all_kmers(s.as_bytes(), &dh, k, base) {
                    fa[h as usize].fetch_add(1, Ordering::Relaxed);
                }
            });
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();
    let freq_atomic = Arc::try_unwrap(freq_atomic).unwrap();
    let freq: Vec<u32> = freq_atomic.into_iter().map(|a| a.into_inner().min(u32::MAX)).collect();
    eprintln!("  Pass 1 done  ({:.1}s)", t0.elapsed().as_secs_f64());

    let mut hash_keys: Vec<u32> = freq.iter().enumerate()
        .filter(|(_, &c)| c as usize > 0 && c as usize <= max_freq)
        .map(|(h, _)| h as u32).collect();
    drop(freq);
    hash_keys.sort_unstable();

    let mut pos_arr = vec![u32::MAX; flat_size];
    for (i, &h) in hash_keys.iter().enumerate() { pos_arr[h as usize] = i as u32; }
    let pos_arr = Arc::new(pos_arr);
    eprintln!("  {} candidate hashes  ({:.1}s)", hash_keys.len(), t0.elapsed().as_secs_f64());

    // ── Pass 1b (1/2): count sketch appearances per hash ─────────────────────
    let n_cand_hashes = hash_keys.len();
    let mut sketch_count: Vec<u32> = vec![0u32; n_cand_hashes];
    done = 0;
    eprintln!("Pass 1b (1/2): counting sketch appearances ...");
    for pq in &seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_strings(&batch, "sequence");
            let pa    = Arc::clone(&pos_arr);
            let dh    = Arc::clone(&dayhoff);
            let batch_sketches: Vec<Vec<u32>> = seqs.par_iter()
                .map(|s| sketch_pos(s.as_bytes(), &dh, k, base, &pa, n_hash))
                .collect();
            for sk in &batch_sketches {
                for &h in sk { sketch_count[pos_arr[h as usize] as usize] += 1; }
            }
            done += batch.num_rows();
            eprint!("  {done:>12}/{total}\r");
        }
    }
    eprintln!();
    let n_seqs = done;

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

    let n_postings: usize = sketch_count.iter().map(|&c| c as usize).sum();
    let mut global_post_offsets: Vec<u64> = Vec::with_capacity(n_hashes + 1);
    let mut offset = 0u64;
    for &c in &sketch_count {
        global_post_offsets.push(offset);
        offset += c as u64;
    }
    global_post_offsets.push(offset);
    drop(sketch_count);

    let n_shards = ((n_postings * 4).saturating_add(SHARD_TARGET_BYTES - 1) / SHARD_TARGET_BYTES).max(1);
    let target_posts_per_shard = (n_postings + n_shards - 1) / n_shards;
    let mut shard_bounds: Vec<usize> = vec![0];
    {
        let mut running = 0usize;
        for i in 0..n_hashes {
            running += (global_post_offsets[i + 1] - global_post_offsets[i]) as usize;
            if running >= target_posts_per_shard && shard_bounds.len() < n_shards {
                shard_bounds.push(i + 1); running = 0;
            }
        }
    }
    shard_bounds.push(n_hashes);
    let actual_shards = shard_bounds.len() - 1;

    eprintln!("Pass 1b (2/2): filling postings ({:.2} GB total, {} shard(s)) ...",
              n_postings as f64 * 4.0 / 1e9, actual_shards);

    let tmp_path = format!("{out_path}.tmp_encoded");
    let tmp_file = std::fs::File::create(&tmp_path).expect("create temp file");
    let mut tmp_writer = BufWriter::with_capacity(64 * 1024 * 1024, tmp_file);
    let mut all_byte_offsets: Vec<u64> = Vec::with_capacity(n_hashes + 1);
    let mut byte_cursor: u64 = 0;
    let mut n_encoded_total: usize = 0;

    for si in 0..actual_shards {
        let sh_start = shard_bounds[si];
        let sh_end   = shard_bounds[si + 1];
        let sh_base  = global_post_offsets[sh_start];
        let sh_posts = (global_post_offsets[sh_end] - sh_base) as usize;
        let shard_post_offsets: Vec<u64> = global_post_offsets[sh_start..=sh_end]
            .iter().map(|&o| o - sh_base).collect();

        eprintln!("  Shard {}/{}: hashes {}..{} ({:.2} GB) ...",
                  si + 1, actual_shards, sh_start, sh_end, sh_posts as f64 * 4.0 / 1e9);

        let mut postings: Vec<u32> = vec![0u32; sh_posts];
        let mut cursors: Vec<u64>  = shard_post_offsets[..sh_end - sh_start].to_vec();
        let mut seq_id = 0u32;
        done = 0;

        for pq in &seq_pqs {
            for batch in open_reader(pq).build().unwrap() {
                let batch = batch.unwrap();
                let seqs  = col_strings(&batch, "sequence");
                let pa    = Arc::clone(&pos_arr2);
                let dh    = Arc::clone(&dayhoff);
                let batch_sketches: Vec<Vec<u32>> = seqs.par_iter()
                    .map(|s| sketch_pos(s.as_bytes(), &dh, k, base, &pa, n_hash))
                    .collect();
                for sk in &batch_sketches {
                    for &h in sk {
                        let pos = pos_arr2[h as usize] as usize;
                        if pos >= sh_start && pos < sh_end {
                            let local = pos - sh_start;
                            postings[cursors[local] as usize] = seq_id;
                            cursors[local] += 1;
                        }
                    }
                    seq_id += 1;
                }
                done += batch.num_rows();
                eprint!("  {done:>12}/{total}\r");
            }
        }
        eprintln!();
        drop(cursors);

        let (shard_offsets, shard_encoded) = encode_shard(&postings, &shard_post_offsets);
        drop(postings);
        for &o in &shard_offsets[..shard_offsets.len() - 1] { all_byte_offsets.push(byte_cursor + o); }
        byte_cursor += *shard_offsets.last().unwrap();
        n_encoded_total += shard_encoded.len();
        tmp_writer.write_all(&shard_encoded).unwrap();
    }
    all_byte_offsets.push(byte_cursor);
    tmp_writer.flush().unwrap();
    drop(tmp_writer);

    eprintln!("  encoded: {:.2} GB  (raw: {:.2} GB, ratio {:.2}x)  ({:.1}s)",
              n_encoded_total as f64 / 1e9, n_postings as f64 * 4.0 / 1e9,
              n_postings as f64 * 4.0 / n_encoded_total as f64, t0.elapsed().as_secs_f64());

    // ── Write final .sidx v4 ──────────────────────────────────────────────────
    eprintln!("Writing {out_path} ...");
    let f = std::fs::File::create(&out_path).expect("create index");
    let mut w = BufWriter::with_capacity(64 * 1024 * 1024, f);

    let mut scheme_field = [0u8; 32];
    let sb = scheme.as_bytes();
    scheme_field[..sb.len().min(32)].copy_from_slice(&sb[..sb.len().min(32)]);

    w.write_all(b"SKETCHIDX4").unwrap();
    w.write_all(&(n_seqs          as u64).to_le_bytes()).unwrap();
    w.write_all(&(n_hashes        as u64).to_le_bytes()).unwrap();
    w.write_all(&(n_encoded_total as u64).to_le_bytes()).unwrap();
    w.write_all(&(max_freq        as u32).to_le_bytes()).unwrap();
    w.write_all(&(n_hash          as u32).to_le_bytes()).unwrap();
    w.write_all(&(k               as u32).to_le_bytes()).unwrap();
    w.write_all(&scheme_field).unwrap();
    for &h in &hash_keys        { w.write_all(&h.to_le_bytes()).unwrap(); }
    for &o in &all_byte_offsets { w.write_all(&o.to_le_bytes()).unwrap(); }

    let mut tmp_r = std::fs::File::open(&tmp_path).expect("open temp file");
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    loop { let n = tmp_r.read(&mut buf).unwrap(); if n == 0 { break; } w.write_all(&buf[..n]).unwrap(); }
    w.flush().unwrap();
    std::fs::remove_file(&tmp_path).ok();

    let sz = std::fs::metadata(&out_path).unwrap().len();
    eprintln!("Done. {:.1}s total  |  {:.3} GB", t0.elapsed().as_secs_f64(), sz as f64 / 1e9);
}
