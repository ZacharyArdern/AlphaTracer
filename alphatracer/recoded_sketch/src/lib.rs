use arrow_array::{Array, StringArray, LargeStringArray};
use memmap2::Mmap;
use needletail::parse_fastx_file;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::io::{BufWriter, Write, Read as IoRead};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub mod parse_pae;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
mod python;

pub const CHUNK: usize = 100_000;

// ── Alphabet ──────────────────────────────────────────────────────────────────

/// Returns (alphabet_table, n_letters) for the named recoding scheme.
/// Valid schemes: murphy2000_4 | murphy2000_5 | murphy2000_8 | dayhoff1978_6
pub fn build_alphabet(scheme: &str) -> ([u8; 256], usize) {
    let mut t = [0xFFu8; 256];
    let n = match scheme {
        "murphy2000_4" => {
            for &c in b"LlVvIiMmCc"     { t[c as usize] = 0; }
            for &c in b"AaGgSsTtPp"     { t[c as usize] = 1; }
            for &c in b"FfYyWw"         { t[c as usize] = 2; }
            for &c in b"EeDdNnQqKkRrHh" { t[c as usize] = 3; }
            4
        }
        "murphy2000_5" => {
            for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
            for &c in b"AaGgSsTtPp" { t[c as usize] = 1; }
            for &c in b"FfYyWw"     { t[c as usize] = 2; }
            for &c in b"EeDdNnQq"   { t[c as usize] = 3; }
            for &c in b"KkRrHh"     { t[c as usize] = 4; }
            5
        }
        "murphy2000_8" => {
            for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
            for &c in b"GgAa"       { t[c as usize] = 1; }
            for &c in b"TtSs"       { t[c as usize] = 2; }
            for &c in b"Pp"         { t[c as usize] = 3; }
            for &c in b"FfYyWw"     { t[c as usize] = 4; }
            for &c in b"EeDdNnQq"   { t[c as usize] = 5; }
            for &c in b"KkRr"       { t[c as usize] = 6; }
            for &c in b"Hh"         { t[c as usize] = 7; }
            8
        }
        "dayhoff1978_6" => {
            for &c in b"IiLlMmVv"   { t[c as usize] = 0; }
            for &c in b"AaGgPpSsTt" { t[c as usize] = 1; }
            for &c in b"FfWwYy"     { t[c as usize] = 2; }
            for &c in b"DdEeNnQq"   { t[c as usize] = 3; }
            for &c in b"HhKkRr"     { t[c as usize] = 4; }
            for &c in b"Cc"         { t[c as usize] = 5; }
            6
        }
        _ => panic!("Unknown recoding scheme '{scheme}'. Valid: murphy2000_4, murphy2000_5, murphy2000_8, dayhoff1978_6"),
    };
    (t, n)
}

// ── K-mer hashing ─────────────────────────────────────────────────────────────

pub fn parse_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}
pub fn parse_u64s(bytes: &[u8]) -> Vec<u64> {
    bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect()
}

pub fn all_kmers(seq: &[u8], dayhoff: &[u8; 256], k: usize, base: u32) -> Vec<u32> {
    let pow_k = base.pow((k - 1) as u32);
    let mut out = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
    let mut seg_start = 0usize;
    for i in 0..=seq.len() {
        if i < seq.len() && dayhoff[seq[i] as usize] != 0xFF { continue; }
        let seg = &seq[seg_start..i];
        if seg.len() >= k {
            let mut h: u32 = 0;
            for j in 0..k { h = h * base + dayhoff[seg[j] as usize] as u32; }
            out.push(h);
            for s in 1..=seg.len() - k {
                h = h.wrapping_sub(pow_k * dayhoff[seg[s-1] as usize] as u32) * base
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

pub fn sketch(seq: &[u8], dayhoff: &[u8; 256], k: usize, n_hash: usize, base: u32) -> Vec<u32> {
    let mut hashes = all_kmers(seq, dayhoff, k, base);
    hashes.truncate(n_hash);
    hashes
}

/// N_HASH smallest k-mers present in pos_arr (u32::MAX = not kept).
pub fn sketch_pos(seq: &[u8], dayhoff: &[u8; 256], k: usize, base: u32,
                  pos_arr: &[u32], n_hash: usize) -> Vec<u32> {
    let kmers = all_kmers(seq, dayhoff, k, base);
    let mut out = Vec::with_capacity(n_hash);
    for &h in &kmers {
        if out.len() == n_hash { break; }
        if pos_arr[h as usize] != u32::MAX { out.push(h); }
    }
    out
}

// ── VarInt ───────────────────────────────────────────────────────────────────

pub fn write_varint(v: u32, out: &mut Vec<u8>) {
    let mut v = v;
    loop {
        if v < 0x80 { out.push(v as u8); break; }
        out.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
}

pub fn read_varint(data: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0u32;
    loop {
        let b = data[*pos];
        *pos += 1;
        v |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 { break; }
        shift += 7;
        if shift >= 32 { break; }
    }
    v
}

// ── Parquet helpers ───────────────────────────────────────────────────────────

pub fn open_reader(path: &str) -> ParquetRecordBatchReaderBuilder<std::fs::File> {
    let file = std::fs::File::open(path).expect("open parquet");
    ParquetRecordBatchReaderBuilder::try_new(file).unwrap().with_batch_size(CHUNK)
}

pub fn total_rows(paths: &[String]) -> usize {
    paths.iter().map(|p| open_reader(p).metadata().file_metadata().num_rows() as usize).sum()
}

pub fn col_strings(batch: &arrow_array::RecordBatch, name: &str) -> Vec<String> {
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

// ── Index ─────────────────────────────────────────────────────────────────────

pub struct Index {
    pub mmap:   Mmap,
    pub inv:    FxHashMap<u32, (u64, u64)>,
    pub n_hash: usize,
    pub n_seqs: usize,
    pub scheme: String,
    pub k:      usize,
}

impl Index {
    pub fn data(&self) -> &[u8] { &self.mmap[..] }
}

pub fn load_index(path: &str) -> Index {
    let file = std::fs::File::open(path).expect("open .sidx");
    let mmap = unsafe { Mmap::map(&file).expect("mmap .sidx") };
    let data: &[u8] = &mmap[..];

    let is_v4 = &data[..10] == b"SKETCHIDX4";
    let mut pos = 10usize;
    macro_rules! read_u32 { () => {{ let v = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4; v }} }
    macro_rules! read_u64 { () => {{ let v = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8; v }} }

    let n_seqs   = read_u64!() as usize;
    let n_hashes = read_u64!() as usize;
    let _        = read_u64!(); // n_encoded
    let _        = read_u32!(); // max_freq
    let n_hash   = read_u32!() as usize;

    let (scheme, k) = if is_v4 {
        let k   = read_u32!() as usize;
        let sb  = &data[pos..pos + 32];
        let len = sb.iter().position(|&b| b == 0).unwrap_or(32);
        let s   = std::str::from_utf8(&sb[..len]).expect("scheme utf8").to_string();
        pos += 32;
        (s, k)
    } else {
        ("murphy2000_5".to_string(), 11usize)
    };

    let hash_keys = parse_u32s(&data[pos..pos + n_hashes * 4]);
    pos += n_hashes * 4;
    let offsets  = parse_u64s(&data[pos..pos + (n_hashes + 1) * 8]);
    let enc_base = (pos + (n_hashes + 1) * 8) as u64;

    let inv: FxHashMap<u32, (u64, u64)> = hash_keys.into_iter()
        .zip(offsets.windows(2))
        .map(|(k, w)| (k, (enc_base + w[0], enc_base + w[1])))
        .collect();

    Index { mmap, inv, n_hash, n_seqs, scheme, k }
}

// ── Search ────────────────────────────────────────────────────────────────────

pub fn top_k_from_hits(mut hits: Vec<u32>, min_shared: u32, top_k: usize) -> Vec<(u32, u32)> {
    hits.sort_unstable();
    let mut heap = BinaryHeap::new();
    for group in hits.chunk_by(|a, b| a == b) {
        let count = group.len() as u32;
        if count >= min_shared { heap.push((count, group[0])); }
    }
    (0..top_k).filter_map(|_| heap.pop()).collect()
}

pub fn run_search(
    idx: &Index,
    query_sketches: &[Vec<u32>],
    min_shared: u32,
    top_k: usize,
    n_hash_search: usize,
    progress: &AtomicUsize,
) -> Vec<Vec<(u32, u32)>> {
    use rayon::prelude::*;
    let data = idx.data();
    let inv  = &idx.inv;
    query_sketches.par_iter().map(|sk| {
        let sk = &sk[..n_hash_search.min(sk.len())];
        let mut ranges: Vec<(u64, u64)> = sk.iter()
            .filter_map(|h| inv.get(h).copied())
            .collect();
        ranges.sort_unstable_by_key(|&(start, _)| start);
        let mut hits: Vec<u32> = Vec::with_capacity(ranges.len() * 64);
        for (start, end) in ranges {
            let mut p = start as usize;
            let mut prev = 0u32;
            while p < end as usize {
                let delta = read_varint(data, &mut p);
                prev += delta;
                hits.push(prev);
            }
        }
        let top = top_k_from_hits(hits, min_shared, top_k);
        progress.fetch_add(1, Ordering::Relaxed);
        top
    }).collect()
}

// ── FASTA reading ─────────────────────────────────────────────────────────────

pub fn read_fasta(path: &str) -> (Vec<String>, Vec<Vec<u8>>) {
    let mut ids:  Vec<String>  = Vec::new();
    let mut seqs: Vec<Vec<u8>> = Vec::new();
    let mut reader = parse_fastx_file(path).expect("open fasta");
    while let Some(rec) = reader.next() {
        let rec = rec.unwrap();
        let id  = String::from_utf8_lossy(rec.id())
                    .split_whitespace().next().unwrap_or("").to_string();
        let seq = rec.seq();
        let seq = if seq.last() == Some(&b'*') { seq[..seq.len()-1].to_vec() }
                  else { seq.to_vec() };
        ids.push(id);
        seqs.push(seq);
    }
    (ids, seqs)
}

// ── Index building ────────────────────────────────────────────────────────────

const SHARD_TARGET_BYTES: usize = 8_000_000_000;

fn encode_shard_internal(postings: &[u32], post_offsets: &[u64]) -> (Vec<u64>, Vec<u8>) {
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

/// Build a .sidx v4 inverted index from one or more sequence parquets.
///
/// * `seq_pqs`      – paths to input parquet files (must have a `sequence` column)
/// * `out_path`     – destination `.sidx` file path
/// * `k`            – k-mer length
/// * `max_freq_arg` – if < 1.0, treated as a fraction of total seqs; >= 1.0 treated as absolute count
/// * `n_hash`       – number of min-hashes per sketch
/// * `scheme`       – recoding scheme name (e.g. `"murphy2000_5"`)
///
/// Returns `Ok(n_seqs)` on success or `Err(message)` on failure.
pub fn build_index(
    seq_pqs: &[String],
    out_path: &str,
    k: usize,
    max_freq_arg: f64,
    n_hash: usize,
    scheme: &str,
) -> Result<usize, String> {
    let (dayhoff_arr, n_letters) = build_alphabet(scheme);
    let base      = n_letters as u32;
    let flat_size = n_letters.pow(k as u32);
    let dayhoff   = Arc::new(dayhoff_arr);
    let t0        = Instant::now();

    let total    = total_rows(seq_pqs);
    let max_freq = if max_freq_arg < 1.0 {
        ((total as f64 * max_freq_arg) as usize).max(1)
    } else {
        max_freq_arg as usize
    };
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
    for pq in seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_strings(&batch, "sequence");
            let dh    = Arc::clone(&dayhoff);
            let fa    = Arc::clone(&freq_atomic);
            use rayon::prelude::*;
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

    // ── Pass 1b (1/2): count sketch appearances per hash ────────────────────
    let n_cand_hashes = hash_keys.len();
    let mut sketch_count: Vec<u32> = vec![0u32; n_cand_hashes];
    done = 0;
    eprintln!("Pass 1b (1/2): counting sketch appearances ...");
    for pq in seq_pqs {
        for batch in open_reader(pq).build().unwrap() {
            let batch = batch.unwrap();
            let seqs  = col_strings(&batch, "sequence");
            let pa    = Arc::clone(&pos_arr);
            let dh    = Arc::clone(&dayhoff);
            use rayon::prelude::*;
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
    let tmp_file = std::fs::File::create(&tmp_path)
        .map_err(|e| format!("create temp file: {e}"))?;
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

        for pq in seq_pqs {
            for batch in open_reader(pq).build().unwrap() {
                let batch = batch.unwrap();
                let seqs  = col_strings(&batch, "sequence");
                let pa    = Arc::clone(&pos_arr2);
                let dh    = Arc::clone(&dayhoff);
                use rayon::prelude::*;
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

        let (shard_offsets, shard_encoded) = encode_shard_internal(&postings, &shard_post_offsets);
        drop(postings);
        for &o in &shard_offsets[..shard_offsets.len() - 1] { all_byte_offsets.push(byte_cursor + o); }
        byte_cursor += *shard_offsets.last().unwrap();
        n_encoded_total += shard_encoded.len();
        tmp_writer.write_all(&shard_encoded)
            .map_err(|e| format!("write temp file: {e}"))?;
    }
    all_byte_offsets.push(byte_cursor);
    tmp_writer.flush().map_err(|e| format!("flush temp file: {e}"))?;
    drop(tmp_writer);

    eprintln!("  encoded: {:.2} GB  (raw: {:.2} GB, ratio {:.2}x)  ({:.1}s)",
              n_encoded_total as f64 / 1e9, n_postings as f64 * 4.0 / 1e9,
              n_postings as f64 * 4.0 / n_encoded_total as f64, t0.elapsed().as_secs_f64());

    // ── Write final .sidx v4 ─────────────────────────────────────────────────
    eprintln!("Writing {out_path} ...");
    let f = std::fs::File::create(out_path)
        .map_err(|e| format!("create index file: {e}"))?;
    let mut w = BufWriter::with_capacity(64 * 1024 * 1024, f);

    let mut scheme_field = [0u8; 32];
    let sb = scheme.as_bytes();
    scheme_field[..sb.len().min(32)].copy_from_slice(&sb[..sb.len().min(32)]);

    w.write_all(b"SKETCHIDX4").map_err(|e| format!("write header: {e}"))?;
    w.write_all(&(n_seqs          as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&(n_hashes        as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&(n_encoded_total as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&(max_freq        as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&(n_hash          as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&(k               as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&scheme_field).map_err(|e| e.to_string())?;
    for &h in &hash_keys        { w.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?; }
    for &o in &all_byte_offsets { w.write_all(&o.to_le_bytes()).map_err(|e| e.to_string())?; }

    let mut tmp_r = std::fs::File::open(&tmp_path)
        .map_err(|e| format!("open temp file for copy: {e}"))?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    loop {
        let n = tmp_r.read(&mut buf).map_err(|e| format!("read temp file: {e}"))?;
        if n == 0 { break; }
        w.write_all(&buf[..n]).map_err(|e| format!("write index: {e}"))?;
    }
    w.flush().map_err(|e| format!("flush index: {e}"))?;
    std::fs::remove_file(&tmp_path).ok();

    let sz = std::fs::metadata(out_path).unwrap_or_else(|_| panic!("stat {out_path}")).len();
    eprintln!("Done. {:.1}s total  |  {:.3} GB", t0.elapsed().as_secs_f64(), sz as f64 / 1e9);

    Ok(n_seqs)
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn alphatracer_sketch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
