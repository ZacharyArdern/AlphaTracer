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

const DEFAULT_K: usize = 11;
const DEFAULT_N_HASH: usize = 64;
const DEFAULT_SCHEME: &str = "murphy2000_5";

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
    let scheme   = scheme_arg.as_deref().unwrap_or(DEFAULT_SCHEME).to_string();

    match recoded_sketch::build_index(&seq_pqs, &out_path, k, mf_arg, n_hash, &scheme) {
        Ok(n_seqs) => eprintln!("Indexed {n_seqs} sequences."),
        Err(e) => { eprintln!("index-seqs error: {e}"); std::process::exit(1); }
    }
}
