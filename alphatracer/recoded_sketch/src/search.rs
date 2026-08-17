use recoded_sketch::{build_alphabet, load_index, run_search, sketch, read_fasta};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const TOP_K: usize = 5;
const MIN_SHARED: u32 = 2;

fn db_label(path: &str) -> String {
    let lower = path.to_lowercase();
    if lower.contains("esm") { return "ESM Atlas".to_string(); }
    if lower.contains("afdb") || lower.contains("alphafold") { return "AFDB".to_string(); }
    std::path::Path::new(path).file_stem()
        .and_then(|s| s.to_str()).unwrap_or(path).to_string()
}

fn spawn_progress_thread(progress: Arc<AtomicUsize>, total: usize, label: String) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let n = progress.load(Ordering::Relaxed);
            if n > 0 {
                let pct = n as f64 / total as f64 * 100.0;
                let filled = (pct / 5.0).round() as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(20 - filled);
                eprint!("  \x1b[34m{label}: {n}/{total} [{bar}] {pct:.0}%\x1b[0m\r");
            }
            if n >= total { break; }
        }
        eprintln!();
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: search <sketch.sidx> <query.fasta> [top_k] [min_shared] [n_hash_search=0]");
        eprintln!("  k and recoding_scheme are read from the index header (SKETCHIDX4).");
        eprintln!("  For legacy SKETCHIDX3 indexes, defaults k=11 scheme=murphy2000_5.");
        std::process::exit(1);
    }
    let db_path    = &args[1];
    let query_fa   = &args[2];
    let top_k      = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(TOP_K);
    let min_shared = args.get(4).and_then(|s| s.parse::<u32>().ok()).unwrap_or(MIN_SHARED);
    let n_hash_search: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let t0 = Instant::now();
    let label = db_label(db_path);
    eprintln!("  [1/2] Loading index: {label} ...");

    let idx = load_index(db_path);
    eprintln!("  [1/2] Index loaded: {label} — {} seqs, n_hash={}, k={}, scheme={}  ({:.1}s)",
              idx.n_seqs, idx.n_hash, idx.k, idx.scheme, t0.elapsed().as_secs_f64());

    let (dayhoff_arr, n_letters) = build_alphabet(&idx.scheme);
    let base          = n_letters as u32;
    let k             = idx.k;
    let n_hash        = idx.n_hash;
    let n_hash_search = if n_hash_search == 0 { n_hash } else { n_hash_search };
    let dayhoff       = Arc::new(dayhoff_arr);

    let (query_ids, query_seqs) = read_fasta(query_fa);
    let n_queries = query_ids.len();
    eprintln!("  [2/2] Searching {n_queries} queries against {label} (top_k={top_k})...");

    let dh = Arc::clone(&dayhoff);
    let query_sketches: Vec<Vec<u32>> = query_seqs.par_iter()
        .map(|s| sketch(s, &dh, k, n_hash, base))
        .collect();

    let t1 = Instant::now();
    let progress = Arc::new(AtomicUsize::new(0));
    spawn_progress_thread(Arc::clone(&progress), n_queries, label.clone());
    let results = run_search(&idx, &query_sketches, min_shared, top_k, n_hash_search, &progress);
    let t_search = t1.elapsed().as_secs_f64();
    eprintln!("Search: {n_queries} queries in {t_search:.3}s  ({:.0} q/s)",
              n_queries as f64 / t_search);

    println!("query\trow_idx\tshared\tcontainment_value");
    for (qi, hits) in results.iter().enumerate() {
        for &(shared, seq_idx) in hits {
            println!("{}\t{}\t{}\t{:.4}", query_ids[qi], seq_idx, shared,
                     shared as f64 / n_hash_search as f64);
        }
    }
}
