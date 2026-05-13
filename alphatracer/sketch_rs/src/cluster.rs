/// cluster: pair-finding + union-find clustering over function+family groups.
///
/// Usage:
///   cluster <seq.parquet> <sketch.parquet> <out.parquet> [threshold=0.17] [min_shared=2] [max_bucket=200]
///
/// Both input parquets must be in the same row order (as produced by fasta_to_parquet.py +
/// the sketch binary). seq.parquet needs columns: AFDB_ID, sequence, function, family.
/// sketch.parquet needs columns: AFDB_ID, sketch (FixedSizeList[100] u32).
///
/// Output columns: rep_AFDB_ID, sequence, function, family, group_size, n_reps  (one row per rep).
use arrow_array::{
    Array, Int32Array, LargeStringArray, RecordBatch, StringArray,
    FixedSizeListArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::time::Instant;

const N_HASH: usize = 100;
const U32_MAX: u32 = u32::MAX;
const READ_BATCH: usize = 100_000;
const GROUP_BUFFER: usize = 8_000; // groups dispatched to rayon at a time

// ── Union-Find ────────────────────────────────────────────────────────────────

struct Dsu {
    parent: Vec<u32>,
    rank:   Vec<u8>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu { parent: (0..n as u32).collect(), rank: vec![0; n] }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }
    fn union(&mut self, x: u32, y: u32) {
        let px = self.find(x);
        let py = self.find(y);
        if px == py { return; }
        match self.rank[px as usize].cmp(&self.rank[py as usize]) {
            std::cmp::Ordering::Less    => self.parent[px as usize] = py,
            std::cmp::Ordering::Greater => self.parent[py as usize] = px,
            std::cmp::Ordering::Equal   => {
                self.parent[py as usize] = px;
                self.rank[px as usize] += 1;
            }
        }
    }
}

// ── Per-group data ────────────────────────────────────────────────────────────

struct Group {
    function: String,
    family:   String,
    ids:      Vec<String>,
    seqs:     Vec<String>,
    lens:     Vec<u32>,
    sketches: Vec<Vec<u32>>, // filtered (no U32_MAX); each sketch already sorted
}

// ── Rep output row ────────────────────────────────────────────────────────────

struct Rep {
    id:         String,
    sequence:   String,
    function:   String,
    family:     String,
    group_size: i32,
    n_reps:     i32,
}

// ── Merge-count intersection of two sorted slices ────────────────────────────

fn count_shared(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal   => { n += 1; i += 1; j += 1; }
            std::cmp::Ordering::Less    => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    n
}

// ── Cluster one group, return reps ───────────────────────────────────────────

fn cluster_group(group: &Group, threshold: f64, min_shared: usize, max_bucket: usize) -> Vec<Rep> {
    let n = group.ids.len();

    // Size-2 fast path: merge-count, no allocation
    if n == 2 {
        let shared = count_shared(&group.sketches[0], &group.sketches[1]);
        let mut rep_indices: Vec<usize> = Vec::new();
        if shared >= min_shared {
            let union = group.sketches[0].len() + group.sketches[1].len() - shared;
            if union > 0 && shared as f64 / union as f64 >= threshold {
                let r = if group.lens[0] >= group.lens[1] { 0 } else { 1 };
                rep_indices.push(r);
            }
        }
        if rep_indices.is_empty() {
            rep_indices.push(0);
            rep_indices.push(1);
        }
        let nr = rep_indices.len() as i32;
        return rep_indices.into_iter().map(|i| Rep {
            id:         group.ids[i].clone(),
            sequence:   group.seqs[i].clone(),
            function:   group.function.clone(),
            family:     group.family.clone(),
            group_size: n as i32,
            n_reps:     nr,
        }).collect();
    }

    // ── Flat-sort pair-finding ────────────────────────────────────────────────
    // Adaptive max_bucket: large groups have large within-group bucket sizes.
    // Cap aggressively to keep FxHashMap bounded (~100M pairs max).
    let eff_max_bucket = if n > 100_000 { 5.min(max_bucket) }
                         else if n > 10_000 { 20.min(max_bucket) }
                         else { max_bucket };
    let total_hashes: usize = group.sketches.iter().map(|s| s.len()).sum();
    let mut flat: Vec<(u32, u32)> = Vec::with_capacity(total_hashes);
    for (i, sk) in group.sketches.iter().enumerate() {
        for &h in sk {
            flat.push((h, i as u32));
        }
    }
    flat.sort_unstable_by_key(|&(h, _)| h);

    // Count shared hashes per pair
    let mut shared_map: FxHashMap<u64, u32> = FxHashMap::default();
    let multiplier = n as u64;
    let mut i = 0;
    while i < flat.len() {
        let h = flat[i].0;
        let mut j = i;
        while j < flat.len() && flat[j].0 == h { j += 1; }
        let bsz = j - i;
        if (2..=eff_max_bucket).contains(&bsz) {
            for a in i..j {
                for b in (a + 1)..j {
                    let mut sa = flat[a].1 as u64;
                    let mut sb = flat[b].1 as u64;
                    if sa > sb { std::mem::swap(&mut sa, &mut sb); }
                    *shared_map.entry(sa * multiplier + sb).or_insert(0) += 1;
                }
            }
        }
        i = j;
    }

    if shared_map.is_empty() {
        // No shared hashes — every seq is its own cluster
        let nr = n as i32;
        return (0..n).map(|i| Rep {
            id:         group.ids[i].clone(),
            sequence:   group.seqs[i].clone(),
            function:   group.function.clone(),
            family:     group.family.clone(),
            group_size: nr,
            n_reps:     nr,
        }).collect();
    }

    // ── Jaccard filter + DSU ─────────────────────────────────────────────────
    let hash_counts: Vec<usize> = group.sketches.iter().map(|s| s.len()).collect();
    let mut dsu = Dsu::new(n);

    for (key, &shared) in &shared_map {
        let s = shared as usize;
        if s < min_shared { continue; }
        let a = (key / multiplier) as usize;
        let b = (key % multiplier) as usize;
        let union = hash_counts[a] + hash_counts[b] - s;
        if union > 0 && s as f64 / union as f64 >= threshold {
            dsu.union(a as u32, b as u32);
        }
    }

    // Pick longest rep per component
    let mut comp_rep: FxHashMap<u32, usize> = FxHashMap::default(); // root → rep_idx
    for i in 0..n {
        let root = dsu.find(i as u32);
        let entry = comp_rep.entry(root).or_insert(i);
        if group.lens[i] > group.lens[*entry] { *entry = i; }
    }

    let nr = comp_rep.len() as i32;
    comp_rep.values().map(|&i| Rep {
        id:         group.ids[i].clone(),
        sequence:   group.seqs[i].clone(),
        function:   group.function.clone(),
        family:     group.family.clone(),
        group_size: n as i32,
        n_reps:     nr,
    }).collect()
}

// ── Parquet helpers ───────────────────────────────────────────────────────────

fn col_strings(batch: &RecordBatch, name: &str) -> Vec<String> {
    let col = batch.column_by_name(name).unwrap_or_else(|| panic!("missing column: {name}"));
    let n = col.len();
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        (0..n).map(|i| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        (0..n).map(|i| if a.is_null(i) { String::new() } else { a.value(i).to_string() }).collect()
    } else {
        vec![String::new(); n]
    }
}

fn open_reader(path: &str) -> ParquetRecordBatchReaderBuilder<std::fs::File> {
    let file = std::fs::File::open(path).unwrap_or_else(|_| panic!("cannot open {path}"));
    ParquetRecordBatchReaderBuilder::try_new(file).unwrap().with_batch_size(READ_BATCH)
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cluster <seq.parquet> <sketch.parquet> <out.parquet> [threshold=0.17] [min_shared=2] [max_bucket=200]");
        std::process::exit(1);
    }

    let seq_path    = &args[1];
    let sketch_path = &args[2];
    let out_path    = &args[3];
    let threshold   = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.17f64);
    let min_shared  = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2usize);
    let max_bucket  = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(200usize);

    eprintln!("cluster: seq={seq_path}  sketch={sketch_path}  out={out_path}");
    eprintln!("  threshold={threshold}  min_shared={min_shared}  max_bucket={max_bucket}");

    // ── Output parquet writer ─────────────────────────────────────────────────
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("rep_AFDB_ID", DataType::LargeUtf8, false),
        Field::new("sequence",    DataType::LargeUtf8, false),
        Field::new("function",    DataType::LargeUtf8, false),
        Field::new("family",      DataType::LargeUtf8, false),
        Field::new("group_size",  DataType::Int32,     false),
        Field::new("n_reps",      DataType::Int32,     false),
    ]));
    let props    = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let out_file = std::fs::File::create(out_path).unwrap();
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props)).unwrap();

    // ── Stream both parquets together ─────────────────────────────────────────
    let t0 = Instant::now();

    let seq_reader    = open_reader(seq_path).build().unwrap();
    let sketch_reader = open_reader(sketch_path).build().unwrap();

    let mut cur_fn  = String::new();
    let mut cur_fam = String::new();
    let mut cur_ids:  Vec<String>   = Vec::new();
    let mut cur_seqs: Vec<String>   = Vec::new();
    let mut cur_lens: Vec<u32>      = Vec::new();
    let mut cur_sks:  Vec<Vec<u32>> = Vec::new();

    let mut group_buf: Vec<Group> = Vec::with_capacity(GROUP_BUFFER);
    let mut total_groups = 0u64;
    let mut total_in     = 0u64;
    let mut total_reps   = 0u64;

    // Write a batch of reps to parquet
    let write_reps = |writer: &mut ArrowWriter<_>, reps: &[Rep], schema: &Arc<Schema>| {
        if reps.is_empty() { return; }
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(LargeStringArray::from(reps.iter().map(|r| r.id.as_str()).collect::<Vec<_>>())),
            Arc::new(LargeStringArray::from(reps.iter().map(|r| r.sequence.as_str()).collect::<Vec<_>>())),
            Arc::new(LargeStringArray::from(reps.iter().map(|r| r.function.as_str()).collect::<Vec<_>>())),
            Arc::new(LargeStringArray::from(reps.iter().map(|r| r.family.as_str()).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(reps.iter().map(|r| r.group_size).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(reps.iter().map(|r| r.n_reps).collect::<Vec<_>>())),
        ]).unwrap();
        writer.write(&batch).unwrap();
    };

    // Flush GROUP_BUFFER groups through rayon, write output
    let flush = |group_buf: &mut Vec<Group>,
                 writer: &mut ArrowWriter<_>,
                 total_groups: &mut u64,
                 total_in: &mut u64,
                 total_reps: &mut u64,
                 schema: &Arc<Schema>,
                 threshold: f64,
                 min_shared: usize,
                 max_bucket: usize,
                 t0: Instant| {
        let all_reps: Vec<Vec<Rep>> = group_buf
            .par_iter()
            .map(|g| cluster_group(g, threshold, min_shared, max_bucket))
            .collect();

        let flat: Vec<Rep> = all_reps.into_iter().flatten().collect();
        *total_groups += group_buf.len() as u64;
        *total_in     += group_buf.iter().map(|g| g.ids.len() as u64).sum::<u64>();
        *total_reps   += flat.len() as u64;

        write_reps(writer, &flat, schema);
        group_buf.clear();

        let elapsed = t0.elapsed().as_secs_f64();
        let rate    = *total_in as f64 / elapsed;
        let reduc   = 100.0 * (1.0 - *total_reps as f64 / (*total_in as f64).max(1.0));
        eprint!(
            "\r  {total_groups:>9} groups | {total_in:>12} → {total_reps:>12} reps \
             ({reduc:5.1}% reduc) | {rate:>9.0} seq/s | {elapsed:>6.0}s"
        );
    };

    // Finish current group and push to buffer (if >1 seq)
    let push_group = |cur_fn: &mut String, cur_fam: &mut String,
                      cur_ids: &mut Vec<String>, cur_seqs: &mut Vec<String>,
                      cur_lens: &mut Vec<u32>, cur_sks: &mut Vec<Vec<u32>>,
                      group_buf: &mut Vec<Group>| {
        if cur_ids.len() > 1 {
            group_buf.push(Group {
                function: cur_fn.clone(),
                family:   cur_fam.clone(),
                ids:      std::mem::take(cur_ids),
                seqs:     std::mem::take(cur_seqs),
                lens:     std::mem::take(cur_lens),
                sketches: std::mem::take(cur_sks),
            });
        } else {
            cur_ids.clear(); cur_seqs.clear(); cur_lens.clear(); cur_sks.clear();
        }
    };

    for (seq_batch_res, sk_batch_res) in seq_reader.zip(sketch_reader) {
        let seq_batch = seq_batch_res.unwrap();
        let sk_batch  = sk_batch_res.unwrap();
        let nr        = seq_batch.num_rows();

        // Verify alignment
        let seq_ids = col_strings(&seq_batch, "AFDB_ID");
        let sk_ids  = col_strings(&sk_batch,  "AFDB_ID");
        assert_eq!(seq_ids[0], sk_ids[0], "Row order mismatch between seq and sketch parquets");

        let fns  = col_strings(&seq_batch, "function");
        let fams = col_strings(&seq_batch, "family");
        let seqs = col_strings(&seq_batch, "sequence");

        // Extract sketch flat values
        let sk_col  = sk_batch.column_by_name("sketch").unwrap();
        let sk_fsl  = sk_col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let sk_vals = sk_fsl.values();
        let sk_u32  = sk_vals.as_any().downcast_ref::<UInt32Array>().unwrap();
        let sk_flat = sk_u32.values(); // flat &[u32], length = nr * N_HASH

        for i in 0..nr {
            let fn_  = fns[i].as_str();
            let fam  = fams[i].as_str();
            if fn_.is_empty() || fam.is_empty() {
                push_group(&mut cur_fn, &mut cur_fam, &mut cur_ids, &mut cur_seqs,
                           &mut cur_lens, &mut cur_sks, &mut group_buf);
                if group_buf.len() >= GROUP_BUFFER {
                    flush(&mut group_buf, &mut writer,
                          &mut total_groups, &mut total_in, &mut total_reps,
                          &out_schema, threshold, min_shared, max_bucket, t0);
                }
                cur_fn.clear(); cur_fam.clear();
                continue;
            }

            if fn_ != cur_fn || fam != cur_fam {
                push_group(&mut cur_fn, &mut cur_fam, &mut cur_ids, &mut cur_seqs,
                           &mut cur_lens, &mut cur_sks, &mut group_buf);
                if group_buf.len() >= GROUP_BUFFER {
                    flush(&mut group_buf, &mut writer,
                          &mut total_groups, &mut total_in, &mut total_reps,
                          &out_schema, threshold, min_shared, max_bucket, t0);
                }
                cur_fn  = fn_.to_string();
                cur_fam = fam.to_string();
            }

            // Sequence (strip trailing '*') and its length
            let seq  = seqs[i].trim_end_matches('*');
            let len_ = seq.len() as u32;

            // Sketch: slice from flat buffer, filter U32_MAX, already sorted
            let base = i * N_HASH;
            let sk: Vec<u32> = sk_flat[base..base + N_HASH]
                .iter()
                .copied()
                .filter(|&h| h != U32_MAX)
                .collect();

            cur_ids.push(seq_ids[i].clone());
            cur_seqs.push(seq.to_string());
            cur_lens.push(len_);
            cur_sks.push(sk);
        }
    }

    // Flush last group + remaining buffer
    push_group(&mut cur_fn, &mut cur_fam, &mut cur_ids, &mut cur_seqs,
               &mut cur_lens, &mut cur_sks, &mut group_buf);
    if !group_buf.is_empty() {
        flush(&mut group_buf, &mut writer,
              &mut total_groups, &mut total_in, &mut total_reps,
              &out_schema, threshold, min_shared, max_bucket, t0);
    }

    writer.close().unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    let reduc   = 100.0 * (1.0 - total_reps as f64 / (total_in as f64).max(1.0));
    eprintln!();
    eprintln!(
        "\nDone. {total_groups} groups | {total_in} → {total_reps} reps \
         ({reduc:.1}% reduction) | {elapsed:.1}s"
    );
}
