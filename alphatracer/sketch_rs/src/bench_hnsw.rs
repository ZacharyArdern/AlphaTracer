/// Benchmark: HNSW retrieval vs AT inverted-index search
///
/// Both use identical Dayhoff k-mer sketches (same k, n_hash).
/// The only difference is the retrieval structure:
///   AT:   inverted index → collect hits → count shared hashes → rank
///   HNSW: L2 distance on sketch vectors → approximate nearest neighbour
///
/// Usage: bench-hnsw <db.fasta> <query.fasta> [k=9] [n_hash=64] [top_k=5]

use hnsw_rs::prelude::*;
use needletail::parse_fastx_file;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::time::Instant;

fn build_dayhoff() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
    for &c in b"AaGgSsTtPp" { t[c as usize] = 1; }
    for &c in b"FfYyWw"     { t[c as usize] = 2; }
    for &c in b"EeDdNnQq"   { t[c as usize] = 3; }
    for &c in b"KkRrHh"     { t[c as usize] = 4; }
    t
}

fn sketch_seq(seq: &[u8], dayhoff: &[u8; 256], k: usize, pow_k: u32, n_hash: usize) -> Vec<u32> {
    let mut hashes = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
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
    // pad with u32::MAX so all sketches are same length (needed for HNSW fixed-dim distance)
    hashes.resize(n_hash, u32::MAX);
    hashes
}

fn read_fasta(path: &str) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut reader = parse_fastx_file(path).expect("open fasta");
    while let Some(rec) = reader.next() {
        let rec = rec.unwrap();
        let id  = String::from_utf8_lossy(rec.id())
                    .split_whitespace().next().unwrap_or("").to_string();
        let seq = rec.seq().to_vec();
        out.push((id, seq));
    }
    out
}

// ── Distance functions ────────────────────────────────────────────────────────

// For HNSW: 1 - Jaccard (treating u32 hashes as a set, ignoring MAX sentinels).
// Lower distance = more similar. Uses L1 on sorted sketch as proxy.
// True Jaccard = shared / n_hash (where shared excludes MAX).
#[derive(Clone)]
struct JaccardDist;

impl Distance<u32> for JaccardDist {
    fn eval(&self, a: &[u32], b: &[u32]) -> f32 {
        let mut shared = 0u32;
        let mut i = 0;
        let mut j = 0;
        while i < a.len() && j < b.len() {
            if a[i] == u32::MAX && b[j] == u32::MAX { break; }
            if a[i] == u32::MAX { break; }
            if b[j] == u32::MAX { break; }
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => { shared += 1; i += 1; j += 1; }
                std::cmp::Ordering::Less  => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        let n = a.iter().filter(|&&x| x != u32::MAX).count().max(1);
        1.0 - (shared as f32 / n as f32)
    }
}

// ── AT inverted index ─────────────────────────────────────────────────────────

fn build_inverted(sketches: &[Vec<u32>]) -> FxHashMap<u32, Vec<u32>> {
    let mut inv: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (i, sk) in sketches.iter().enumerate() {
        for &h in sk {
            if h != u32::MAX {
                inv.entry(h).or_default().push(i as u32);
            }
        }
    }
    inv
}

fn search_inverted(
    inv: &FxHashMap<u32, Vec<u32>>,
    query_sketches: &[Vec<u32>],
    top_k: usize,
) -> Vec<Vec<(u32, u32)>> {
    query_sketches.par_iter().map(|sk| {
        let mut hits: Vec<u32> = Vec::with_capacity(sk.len() * 32);
        for &h in sk {
            if h == u32::MAX { break; }
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
            heap.push((count, seq_idx));
        }
        let mut top = Vec::with_capacity(top_k);
        while top.len() < top_k { match heap.pop() { Some(x) => top.push(x), None => break } }
        top
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: bench-hnsw <db.fasta> <query.fasta> [k=9] [n_hash=64] [top_k=5]");
        std::process::exit(1);
    }
    let db_path    = &args[1];
    let query_path = &args[2];
    let k          = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(9usize);
    let n_hash     = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let top_k      = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5usize);

    let dayhoff = build_dayhoff();
    let pow_k   = 5u32.pow((k - 1) as u32);

    eprintln!("Reading DB: {db_path}");
    let t0 = Instant::now();
    let db = read_fasta(db_path);
    let n_db = db.len();
    eprintln!("  {n_db} sequences  ({:.2}s)", t0.elapsed().as_secs_f64());

    eprintln!("Reading queries: {query_path}");
    let queries = read_fasta(query_path);
    let n_q = queries.len();
    eprintln!("  {n_q} queries  ({:.2}s)", t0.elapsed().as_secs_f64());

    // ── Sketch everything ─────────────────────────────────────────────────────
    eprintln!("\nSketching DB (k={k}, n_hash={n_hash}) ...");
    let t1 = Instant::now();
    let db_sketches: Vec<Vec<u32>> = db.par_iter()
        .map(|(_, s)| sketch_seq(s, &dayhoff, k, pow_k, n_hash))
        .collect();
    let db_sketch_secs = t1.elapsed().as_secs_f64();
    eprintln!("  {:.2}s  ({:.0} seq/s)", db_sketch_secs, n_db as f64 / db_sketch_secs);

    let t1 = Instant::now();
    let query_sketches: Vec<Vec<u32>> = queries.par_iter()
        .map(|(_, s)| sketch_seq(s, &dayhoff, k, pow_k, n_hash))
        .collect();
    eprintln!("  queries sketched  ({:.2}s)", t1.elapsed().as_secs_f64());

    // ── Build AT inverted index ───────────────────────────────────────────────
    eprintln!("\n── AT inverted index ────────────────────────────────────────────");
    let t1 = Instant::now();
    let inv = build_inverted(&db_sketches);
    let inv_build_secs = t1.elapsed().as_secs_f64();
    eprintln!("  Build: {inv_build_secs:.3}s  ({} unique hashes)", inv.len());

    let t1 = Instant::now();
    let _inv_results = search_inverted(&inv, &query_sketches, top_k);
    let inv_search_secs = t1.elapsed().as_secs_f64();
    eprintln!("  Search: {n_q} queries in {inv_search_secs:.3}s  ({:.0} q/s)",
              n_q as f64 / inv_search_secs);

    // ── Build HNSW index ──────────────────────────────────────────────────────
    eprintln!("\n── HNSW (hnsw_rs, Jaccard distance) ────────────────────────────");
    // ef_construction=400, max_nb_connection=16 are gsearch defaults
    let hnsw: Hnsw<u32, JaccardDist> = Hnsw::new(16, n_db, 16, 400, JaccardDist);

    let t1 = Instant::now();
    let data_for_insert: Vec<(&Vec<u32>, usize)> = db_sketches.iter()
        .enumerate()
        .map(|(i, sk)| (sk, i))
        .collect();
    hnsw.parallel_insert(&data_for_insert);
    let hnsw_build_secs = t1.elapsed().as_secs_f64();
    eprintln!("  Build: {hnsw_build_secs:.3}s  ({:.0} seq/s)", n_db as f64 / hnsw_build_secs);

    // HNSW search
    let ef_search = 64usize;
    let t1 = Instant::now();
    let _hnsw_results: Vec<_> = query_sketches.par_iter().map(|sk| {
        hnsw.search(sk.as_slice(), top_k, ef_search)
    }).collect();
    let hnsw_search_secs = t1.elapsed().as_secs_f64();
    eprintln!("  Search: {n_q} queries in {hnsw_search_secs:.3}s  ({:.0} q/s)",
              n_q as f64 / hnsw_search_secs);

    // ── Summary ───────────────────────────────────────────────────────────────
    eprintln!("\n── Summary ({n_db} DB seqs, {n_q} queries) ──────────────────────");
    eprintln!("              {:>12}  {:>12}", "build (s)", "search (q/s)");
    eprintln!("  AT inv-idx  {:>12.3}  {:>12.0}", inv_build_secs,   n_q as f64 / inv_search_secs);
    eprintln!("  HNSW        {:>12.3}  {:>12.0}", hnsw_build_secs,  n_q as f64 / hnsw_search_secs);
    let speedup = (n_q as f64 / hnsw_search_secs) / (n_q as f64 / inv_search_secs);
    eprintln!("  HNSW search speedup: {speedup:.2}×");
}
