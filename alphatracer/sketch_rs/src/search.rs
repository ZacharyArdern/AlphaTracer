use arrow_array::{Array, StringArray, LargeStringArray, FixedSizeListArray, UInt32Array};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use memmap2::Mmap;
use needletail::parse_fastx_file;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

const N_HASH: usize = 100; // v1 default; overridden by v2 header
const CHUNK: usize = 100_000;
const DEFAULT_K: usize = 9;
const MAX_FREQ: usize = 500;
const TOP_K: usize = 5;
const MIN_SHARED: u32 = 2;

fn build_dayhoff() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
    for &c in b"AaGgSsTtPp" { t[c as usize] = 1; }
    for &c in b"FfYyWw"     { t[c as usize] = 2; }
    for &c in b"EeDdNnQq"   { t[c as usize] = 3; }
    for &c in b"KkRrHh"     { t[c as usize] = 4; }
    t
}

fn sketch_seq(seq: &[u8], dayhoff: &[u8; 256], k: usize, n_hash: usize) -> Vec<u32> {
    let pow_k = 5u32.pow((k - 1) as u32);
    let mut hashes: Vec<u32> = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
    let mut seg_start = 0usize;
    for i in 0..=seq.len() {
        if i < seq.len() && dayhoff[seq[i] as usize] != 0xFF { continue; }
        let seg = &seq[seg_start..i];
        if seg.len() >= k {
            let mut h: u32 = 0;
            for j in 0..k { h = h * 5 + dayhoff[seg[j] as usize] as u32; }
            hashes.push(h);
            for s in 1..=seg.len() - k {
                h = h.wrapping_sub(pow_k * dayhoff[seg[s-1] as usize] as u32) * 5
                    + dayhoff[seg[s+k-1] as usize] as u32;
                hashes.push(h);
            }
        }
        seg_start = i + 1;
    }
    hashes.sort_unstable();
    hashes.dedup();
    hashes.truncate(n_hash);
    hashes
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

// ── Index types ───────────────────────────────────────────────────────────────

// V2: mmap-based lazy loading — only pages actually touched are read from disk.
// hash → (byte_start, byte_end) into the mmap'd file buffer.
struct V2Index {
    mmap:           Mmap,
    inv:            FxHashMap<u32, (u64, u64)>,
    n_hash:         usize,
    n_seqs:         usize,
    id_lens_offset: usize, // byte offset in mmap where id_lengths u32[] starts
    id_bytes_start: usize, // byte offset in mmap where id UTF-8 bytes start
}

impl V2Index {
    fn data(&self) -> &[u8] { &self.mmap[..] }

    // Scan id_lens once and collect IDs for the given seq_idx values.
    // needed need not be sorted; returns a map from seq_idx → id string.
    fn lookup_ids(&self, needed: &[u32]) -> FxHashMap<u32, String> {
        if needed.is_empty() { return FxHashMap::default(); }
        let data = self.data();
        let id_lens: &[u32] = unsafe {
            std::slice::from_raw_parts(
                data[self.id_lens_offset..].as_ptr() as *const u32,
                self.n_seqs,
            )
        };
        let needed_set: FxHashMap<u32, ()> = needed.iter().map(|&i| (i, ())).collect();
        let mut result: FxHashMap<u32, String> = FxHashMap::with_capacity_and_hasher(
            needed.len(), Default::default()
        );
        let mut byte_off = 0usize;
        for (i, &len) in id_lens.iter().enumerate() {
            if needed_set.contains_key(&(i as u32)) {
                let s = &data[self.id_bytes_start + byte_off
                              ..self.id_bytes_start + byte_off + len as usize];
                result.insert(i as u32, std::str::from_utf8(s).unwrap().to_string());
                if result.len() == needed.len() { break; }
            }
            byte_off += len as usize;
        }
        result
    }
}

fn read_varint(data: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0u32;
    loop {
        let b = data[*pos];
        *pos += 1;
        v |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 { break; }
        shift += 7;
    }
    v
}

fn load_sidx_v2(path: &str) -> V2Index {
    let file = std::fs::File::open(path).expect("open .sidx");
    let mmap = unsafe { Mmap::map(&file).expect("mmap .sidx") };
    let data: &[u8] = &mmap[..];

    let mut pos = 10usize; // skip "SKETCHIDX2"
    macro_rules! read_u32 { () => {{ let v = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4; v }} }
    macro_rules! read_u64 { () => {{ let v = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8; v }} }

    let n_seqs    = read_u64!() as usize;
    let n_hashes  = read_u64!() as usize;
    let n_encoded = read_u64!() as usize;
    let _max_freq = read_u32!();
    let n_hash    = read_u32!() as usize;

    // hash_keys — zero-copy slice into mmap'd data
    let hk_start = pos;
    pos += n_hashes * 4;
    let hash_keys: &[u32] = unsafe {
        std::slice::from_raw_parts(data[hk_start..].as_ptr() as *const u32, n_hashes)
    };

    // byte offsets — zero-copy slice
    let off_start = pos;
    pos += (n_hashes + 1) * 8;
    let offsets: &[u64] = unsafe {
        std::slice::from_raw_parts(data[off_start..].as_ptr() as *const u64, n_hashes + 1)
    };

    // encoded block starts here; offsets are relative to enc_base
    let enc_base = pos as u64;
    pos += n_encoded;

    // Map: hash → absolute byte range in `data` for lazy VarInt decode
    let mut inv: FxHashMap<u32, (u64, u64)> = FxHashMap::with_capacity_and_hasher(
        n_hashes, Default::default()
    );
    for i in 0..n_hashes {
        let start = enc_base + offsets[i];
        let end   = enc_base + offsets[i + 1];
        inv.insert(hash_keys[i], (start, end));
    }

    // Record where IDs start; don't materialize them yet — only looked up after search.
    let id_lens_offset = pos;
    let id_bytes_start = pos + n_seqs * 4;

    V2Index { mmap, inv, n_hash, n_seqs, id_lens_offset, id_bytes_start }
}

fn load_sidx_v1(path: &str) -> (Vec<String>, FxHashMap<u32, Vec<u32>>) {
    let raw = std::fs::read(path).expect("read .sidx");
    let data: &'static [u8] = Box::leak(raw.into_boxed_slice());

    let mut pos = 9usize;
    macro_rules! read_u32 { () => {{ let v = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4; v }} }
    macro_rules! read_u64 { () => {{ let v = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8; v }} }

    let n_seqs     = read_u64!() as usize;
    let n_hashes   = read_u64!() as usize;
    let n_postings = read_u64!() as usize;
    let _max_freq  = read_u32!();
    let _n_hash_c  = read_u32!();

    let hk_start = pos; pos += n_hashes * 4;
    let hash_keys: &[u32] = unsafe { std::slice::from_raw_parts(data[hk_start..].as_ptr() as *const u32, n_hashes) };

    let off_start = pos; pos += (n_hashes + 1) * 8;
    let offsets: &[u64] = unsafe { std::slice::from_raw_parts(data[off_start..].as_ptr() as *const u64, n_hashes + 1) };

    let post_start = pos; pos += n_postings * 4;
    let postings_raw: &[u32] = unsafe { std::slice::from_raw_parts(data[post_start..].as_ptr() as *const u32, n_postings) };

    let mut inverted: FxHashMap<u32, Vec<u32>> = FxHashMap::with_capacity_and_hasher(n_hashes, Default::default());
    for i in 0..n_hashes {
        inverted.insert(hash_keys[i], postings_raw[offsets[i] as usize..offsets[i+1] as usize].to_vec());
    }

    let id_lens: &[u32] = unsafe { std::slice::from_raw_parts(data[pos..].as_ptr() as *const u32, n_seqs) };
    pos += n_seqs * 4;
    let mut db_ids = Vec::with_capacity(n_seqs);
    for &len in id_lens {
        db_ids.push(std::str::from_utf8(&data[pos..pos + len as usize]).unwrap().to_string());
        pos += len as usize;
    }
    (db_ids, inverted)
}

fn load_parquet(path: &str, max_freq: usize) -> (Vec<String>, FxHashMap<u32, Vec<u32>>) {
    let file    = std::fs::File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().with_batch_size(CHUNK);
    let n_rows  = builder.metadata().file_metadata().num_rows() as usize;
    let reader  = builder.build().unwrap();

    let mut db_ids: Vec<String> = Vec::with_capacity(n_rows);
    let mut inverted: FxHashMap<u32, Vec<u32>> = FxHashMap::default();

    for batch in reader {
        let batch    = batch.unwrap();
        let n        = batch.num_rows();
        let base_idx = db_ids.len() as u32;
        db_ids.extend(col_strings(&batch, "AFDB_ID"));

        let sk_col = batch.column_by_name("sketch").unwrap();
        if let Some(fsl) = sk_col.as_any().downcast_ref::<FixedSizeListArray>() {
            let values = fsl.values().as_any().downcast_ref::<UInt32Array>().unwrap();
            let stride = fsl.value_length() as usize;
            for i in 0..n {
                let seq_idx = base_idx + i as u32;
                let start   = i * stride;
                for j in 0..stride {
                    let h = values.value(start + j);
                    if h != u32::MAX { inverted.entry(h).or_default().push(seq_idx); }
                }
            }
        } else {
            let list_arr = sk_col.as_list::<i32>();
            for i in 0..n {
                let seq_idx  = base_idx + i as u32;
                let values   = list_arr.value(i);
                let uint_arr = values.as_primitive::<UInt32Type>();
                for j in 0..uint_arr.len() {
                    let h = uint_arr.value(j);
                    if h != u32::MAX { inverted.entry(h).or_default().push(seq_idx); }
                }
            }
        }
    }
    inverted.retain(|_, v| v.len() <= max_freq);
    (db_ids, inverted)
}

// ── Search ────────────────────────────────────────────────────────────────────

fn search_v2(
    idx: &V2Index,
    query_sketches: &[Vec<u32>],
    min_shared: u32,
    top_k: usize,
    n_hash_search: usize,
) -> Vec<Vec<(u32, u32)>> {
    let data = idx.data();
    let inv  = &idx.inv;
    query_sketches.par_iter().map(|sk| {
        let sk = &sk[..n_hash_search.min(sk.len())];
        let mut hits: Vec<u32> = Vec::with_capacity(sk.len() * 64);
        for &h in sk {
            if let Some(&(start, end)) = inv.get(&h) {
                let mut p = start as usize;
                let end   = end as usize;
                let mut prev = 0u32;
                while p < end {
                    let delta = read_varint(data, &mut p);
                    prev += delta;
                    hits.push(prev);
                }
            }
        }
        hits.sort_unstable();
        let mut heap = BinaryHeap::new();
        let mut i = 0;
        while i < hits.len() {
            let seq_idx = hits[i];
            let mut count = 0u32;
            while i < hits.len() && hits[i] == seq_idx { count += 1; i += 1; }
            if count >= min_shared { heap.push((count, seq_idx)); }
        }
        let mut top = Vec::with_capacity(top_k);
        while top.len() < top_k { match heap.pop() { Some(x) => top.push(x), None => break } }
        top
    }).collect()
}

fn search_eager(
    inv: &Arc<FxHashMap<u32, Vec<u32>>>,
    query_sketches: &[Vec<u32>],
    min_shared: u32,
    top_k: usize,
    n_hash_search: usize,
) -> Vec<Vec<(u32, u32)>> {
    query_sketches.par_iter().map(|sk| {
        let sk = &sk[..n_hash_search.min(sk.len())];
        let mut hits: Vec<u32> = Vec::with_capacity(sk.len() * 64);
        for &h in sk {
            if let Some(posting) = inv.get(&h) {
                hits.extend_from_slice(posting);
            }
        }
        hits.sort_unstable();
        let mut heap = BinaryHeap::new();
        let mut i = 0;
        while i < hits.len() {
            let seq_idx = hits[i];
            let mut count = 0u32;
            while i < hits.len() && hits[i] == seq_idx { count += 1; i += 1; }
            if count >= min_shared { heap.push((count, seq_idx)); }
        }
        let mut top = Vec::with_capacity(top_k);
        while top.len() < top_k { match heap.pop() { Some(x) => top.push(x), None => break } }
        top
    }).collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: search <sketch.parquet|sketch.sidx> <query.fasta> [top_k] [min_shared] [max_freq] [k=9]");
        std::process::exit(1);
    }
    let db_path    = &args[1];
    let query_fa   = &args[2];
    let top_k          = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(TOP_K);
    let min_shared     = args.get(4).and_then(|s| s.parse::<u32>().ok()).unwrap_or(MIN_SHARED);
    let max_freq       = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(MAX_FREQ);
    let k              = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_K);
    // 0 = use all hashes (default); any other value caps the search to that many hashes
    let n_hash_search: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);

    let dayhoff = Arc::new(build_dayhoff());
    let t0 = Instant::now();
    eprintln!("Loading index from {db_path} ...");

    // Detect format
    let is_sidx = db_path.ends_with(".sidx");
    let is_v2 = is_sidx && {
        let mut f = std::fs::File::open(db_path).expect("open");
        use std::io::Read;
        let mut magic = [0u8; 10];
        f.read_exact(&mut magic).unwrap_or(());
        &magic == b"SKETCHIDX2"
    };

    // Load query FASTA before heavy index loading (fast)
    let mut query_ids:  Vec<String>  = Vec::new();
    let mut query_seqs: Vec<Vec<u8>> = Vec::new();
    {
        let mut reader = parse_fastx_file(query_fa).expect("open fasta");
        while let Some(rec) = reader.next() {
            let rec = rec.unwrap();
            let id  = String::from_utf8_lossy(rec.id())
                        .split_whitespace().next().unwrap_or("").to_string();
            let seq = rec.seq();
            let seq = if seq.last() == Some(&b'*') { seq[..seq.len()-1].to_vec() }
                      else { seq.to_vec() };
            query_ids.push(id);
            query_seqs.push(seq);
        }
    }
    let n_queries = query_ids.len();

    if is_v2 {
        // ── V2 lazy path ──────────────────────────────────────────────────────
        let idx = load_sidx_v2(db_path);
        eprintln!("Index loaded: {} seqs, {} hashes, n_hash={}  ({:.1}s)",
                  idx.n_seqs, idx.inv.len(), idx.n_hash, t0.elapsed().as_secs_f64());

        eprintln!("Queries: {n_queries}");
        let n_hash = idx.n_hash;
        let n_hash_search = if n_hash_search == 0 { n_hash } else { n_hash_search };
        eprintln!("k={k}, n_hash={n_hash}, n_hash_search={n_hash_search}, min_shared={min_shared}, top_k={top_k}");

        let dh = Arc::clone(&dayhoff);
        let query_sketches: Vec<Vec<u32>> = query_seqs.par_iter()
            .map(|s| sketch_seq(s, &dh, k, n_hash))
            .collect();

        let t1 = Instant::now();
        let results = search_v2(&idx, &query_sketches, min_shared, top_k, n_hash_search);
        let t_search = t1.elapsed().as_secs_f64();
        eprintln!("Search: {n_queries} queries in {t_search:.3}s  ({:.0} q/s)",
                  n_queries as f64 / t_search);

        let needed: Vec<u32> = results.iter()
            .flat_map(|hits| hits.iter().map(|&(_, si)| si))
            .collect();
        let id_map = idx.lookup_ids(&needed);

        println!("query\ttarget\tshared\tjaccard");
        for (qi, hits) in results.iter().enumerate() {
            for &(shared, seq_idx) in hits {
                let jaccard = shared as f64 / (n_hash * 2 - shared as usize) as f64;
                let target = id_map.get(&seq_idx).map(|s| s.as_str()).unwrap_or("?");
                println!("{}\t{}\t{}\t{:.4}", query_ids[qi], target, shared, jaccard);
            }
        }
    } else {
        // ── Eager path (v1 sidx or parquet) ───────────────────────────────────
        let (db_ids, inv, n_hash) = if is_sidx {
            let (db_ids, inv) = load_sidx_v1(db_path);
            eprintln!("Index loaded: {} seqs, {} hashes  ({:.1}s)",
                      db_ids.len(), inv.len(), t0.elapsed().as_secs_f64());
            (db_ids, inv, N_HASH)
        } else {
            let (db_ids, inv) = load_parquet(db_path, max_freq);
            eprintln!("Index: {} seqs, {} hashes  ({:.1}s)",
                      db_ids.len(), inv.len(), t0.elapsed().as_secs_f64());
            (db_ids, inv, N_HASH)
        };

        eprintln!("Queries: {n_queries}");
        let n_hash_search = if n_hash_search == 0 { n_hash } else { n_hash_search };
        eprintln!("k={k}, n_hash={n_hash}, n_hash_search={n_hash_search}, min_shared={min_shared}, top_k={top_k}");

        let dh = Arc::clone(&dayhoff);
        let query_sketches: Vec<Vec<u32>> = query_seqs.par_iter()
            .map(|s| sketch_seq(s, &dh, k, n_hash))
            .collect();

        let t1 = Instant::now();
        let inv = Arc::new(inv);
        let ids = Arc::new(db_ids);
        let results = search_eager(&inv, &query_sketches, min_shared, top_k, n_hash_search);
        let t_search = t1.elapsed().as_secs_f64();
        eprintln!("Search: {n_queries} queries in {t_search:.3}s  ({:.0} q/s)",
                  n_queries as f64 / t_search);

        println!("query\ttarget\tshared\tjaccard");
        for (qi, hits) in results.iter().enumerate() {
            for &(shared, seq_idx) in hits {
                let jaccard = shared as f64 / (n_hash * 2 - shared as usize) as f64;
                println!("{}\t{}\t{}\t{:.4}", query_ids[qi], ids[seq_idx as usize], shared, jaccard);
            }
        }
    }
}
