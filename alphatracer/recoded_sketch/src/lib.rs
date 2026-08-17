use arrow_array::{Array, StringArray, LargeStringArray};
use memmap2::Mmap;
use needletail::parse_fastx_file;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod utils;

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

#[cfg(feature = "extension-module")]
#[pymodule]
fn alphatracer_sketch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
