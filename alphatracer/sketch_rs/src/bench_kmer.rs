/// Benchmark: AT k-mer sketching vs kmerutils
///
/// AT:        Dayhoff 5-group reduction → base-5 rolling hash → u32 → sort → bottom-N
/// kmerutils: full 20-AA alphabet → 5-bit packing → u64 → sort → bottom-N
///
/// Both produce bottom-k MinHash sketches. Alphabet difference is intentional:
/// AT's Dayhoff reduction improves sensitivity at the cost of specificity.
/// kmerutils k=9 (full AA) ≈ AT k=6 (Dayhoff) in terms of k-mer space size.
///
/// Usage: bench-kmer [k=9] [n_hash=64] [n_seqs=10000] [seq_len=300]

use std::time::Instant;
use kmerutils::aautils::kmeraa::{KmerAA32bit, KmerAA64bit, KmerSeqIterator, KmerSeqIteratorT, SequenceAA};
use kmerutils::base::kmertraits::CompressedKmerT;

// Standard 20-AA alphabet (kmerutils accepts these)
const AA20: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

fn gen_sequences(n: usize, len: usize, mut seed: u64) -> Vec<Vec<u8>> {
    (0..n).map(|_| {
        (0..len).map(|_| {
            // xorshift64
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            AA20[(seed % 20) as usize]
        }).collect()
    }).collect()
}

// ── AT approach ───────────────────────────────────────────────────────────────

fn build_dayhoff() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    for &c in b"LlVvIiMmCc" { t[c as usize] = 0; }
    for &c in b"AaGgSsTtPp" { t[c as usize] = 1; }
    for &c in b"FfYyWw"     { t[c as usize] = 2; }
    for &c in b"EeDdNnQq"   { t[c as usize] = 3; }
    for &c in b"KkRrHh"     { t[c as usize] = 4; }
    t
}

fn at_sketch(seq: &[u8], dayhoff: &[u8; 256], k: usize, pow_k: u32, n_hash: usize) -> Vec<u32> {
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
    hashes
}

// ── kmerutils approach (full 20-AA, u32, k≤6) ────────────────────────────────

fn ku_sketch_u32(seq: &[u8], k: usize, n_hash: usize) -> Vec<u32> {
    let seq_aa = SequenceAA::new(seq);
    let mut iter = KmerSeqIterator::<KmerAA32bit>::new(k, &seq_aa);
    iter.set_range(0, seq_aa.size()).unwrap();
    let mut hashes = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
    while let Some(kmer) = iter.next() {
        let h: u32 = kmer.get_compressed_value();
        hashes.push(h);
    }
    hashes.sort_unstable();
    hashes.dedup();
    hashes.truncate(n_hash);
    hashes
}

// ── kmerutils approach (full 20-AA, u64, k≤12) ───────────────────────────────

fn ku_sketch_u64(seq: &[u8], k: usize, n_hash: usize) -> Vec<u64> {
    let seq_aa = SequenceAA::new(seq);
    let mut iter = KmerSeqIterator::<KmerAA64bit>::new(k, &seq_aa);
    iter.set_range(0, seq_aa.size()).unwrap();
    let mut hashes = Vec::with_capacity(seq.len().saturating_sub(k) + 1);
    while let Some(kmer) = iter.next() {
        let h: u64 = kmer.get_compressed_value();
        hashes.push(h);
    }
    hashes.sort_unstable();
    hashes.dedup();
    hashes.truncate(n_hash);
    hashes
}

fn bench<T>(label: &str, seqs: &[Vec<u8>], f: impl Fn(&[u8]) -> Vec<T>) -> (f64, f64) {
    let total_residues: usize = seqs.iter().map(|s| s.len()).sum();
    // warmup
    for s in seqs.iter().take(200) { std::hint::black_box(f(s)); }
    let t = Instant::now();
    let mut total_hashes = 0usize;
    for s in seqs { total_hashes += std::hint::black_box(f(s)).len(); }
    let secs = t.elapsed().as_secs_f64();
    let seqs_per_s = seqs.len() as f64 / secs;
    let mres_per_s = total_residues as f64 / secs / 1e6;
    let avg_sketch = total_hashes as f64 / seqs.len() as f64;
    eprintln!("{label}");
    eprintln!("  {:.3}s  |  {:.0} seqs/s  |  {:.0} M res/s  |  avg sketch {:.1}",
        secs, seqs_per_s, mres_per_s, avg_sketch);
    (seqs_per_s, mres_per_s)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let k       = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9usize);
    let n_hash  = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let n_seqs  = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000usize);
    let seq_len = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(300usize);

    eprintln!("k={k}  n_hash={n_hash}  n_seqs={n_seqs}  seq_len={seq_len}");
    eprintln!("AT hash space:        5^{k} = {} (Dayhoff 5-group, u32)",
              5usize.pow(k as u32));
    eprintln!("kmerutils hash space: 20^{k} = {} (full AA, u64)\n",
              20usize.pow(k as u32));

    let seqs = gen_sequences(n_seqs, seq_len, 42);

    // ── AT ────────────────────────────────────────────────────────────────────
    let dayhoff = build_dayhoff();
    let pow_k   = 5u32.pow((k - 1) as u32);
    let (at_sq, _) = bench(
        &format!("AT  (Dayhoff-5, base-5 rolling hash, k={k}, u32):"),
        &seqs,
        |s| at_sketch(s, &dayhoff, k, pow_k, n_hash),
    );

    // ── kmerutils k≤6 → u32 ──────────────────────────────────────────────────
    if k <= 6 {
        let (ku_sq, _) = bench(
            &format!("kmerutils (full 20-AA, KmerAA32bit, k={k}, u32):"),
            &seqs,
            |s| ku_sketch_u32(s, k, n_hash),
        );
        eprintln!("\nSpeedup: AT is {:.2}× {} than kmerutils (u32)",
            if at_sq > ku_sq { at_sq / ku_sq } else { ku_sq / at_sq },
            if at_sq > ku_sq { "faster" } else { "slower" });
    }

    // ── kmerutils k≤12 → u64 ─────────────────────────────────────────────────
    let (ku64_sq, _) = bench(
        &format!("kmerutils (full 20-AA, KmerAA64bit, k={k}, u64):"),
        &seqs,
        |s| ku_sketch_u64(s, k, n_hash),
    );
    eprintln!("\nSpeedup AT vs kmerutils (u64): AT is {:.2}× {}",
        if at_sq > ku64_sq { at_sq / ku64_sq } else { ku64_sq / at_sq },
        if at_sq > ku64_sq { "faster" } else { "slower" });
}
