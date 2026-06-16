/// parse_pae  —  batch sparse PAE JSON parser
///
/// Usage: parse_pae <step> <pae_json_path> [<pae_json_path> ...]
///
/// Reads one or more AlphaFold PAE JSON files (single-line format:
///   [{"predicted_aligned_error":[[v,v,...],[v,v,...],...],...}]
/// ) and for each outputs a subsampled (every `step`-th row and column)
/// float32 matrix as raw little-endian bytes to stdout.
///
/// Each matrix is preceded by two u32 values: [n_full, n_out], so the
/// caller knows both the original size and the output size.
///
/// Protocol per file:
///   [u32 n_full LE][u32 n_out LE][f32 × n_out × n_out LE]
///
/// Python reads it as:
///   buf = result.stdout
///   pos = 0
///   matrices = {}
///   for path in paths:
///       n_full, n_out = struct.unpack_from('<II', buf, pos); pos += 8
///       mat = np.frombuffer(buf, dtype=np.float32,
///                           count=n_out*n_out, offset=pos).reshape(n_out, n_out)
///       pos += n_out * n_out * 4
///       matrices[path] = (n_full, mat)

use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: parse_pae <step> <pae_json_path> [<pae_json_path> ...]");
        std::process::exit(1);
    }
    let step: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Invalid step: {}", args[1]);
        std::process::exit(1);
    });
    let paths = &args[2..];

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for path in paths {
        let raw = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to read {}: {}", path, e);
                std::process::exit(1);
            }
        };

        let inner_start = match find_inner_start(&raw) {
            Some(i) => i,
            None => { eprintln!("No [[ in {}", path); std::process::exit(1); }
        };
        let inner_end = match find_inner_end(&raw, inner_start) {
            Some(i) => i,
            None => { eprintln!("No ]] in {}", path); std::process::exit(1); }
        };

        let inner = &raw[inner_start..inner_end];
        let rows = split_rows(inner);
        let n = rows.len();
        let n_out = (n + step - 1) / step;

        let mut values: Vec<f32> = Vec::with_capacity(n_out * n_out);
        for (row_i, row_bytes) in rows.iter().enumerate() {
            if row_i % step == 0 {
                parse_row_sparse(row_bytes, step, n, &mut values);
            }
        }

        out.write_all(&(n as u32).to_le_bytes()).unwrap();
        out.write_all(&(n_out as u32).to_le_bytes()).unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        };
        out.write_all(bytes).unwrap();
    }
}

fn find_inner_start(raw: &[u8]) -> Option<usize> {
    for i in 0..raw.len().saturating_sub(1) {
        if raw[i] == b'[' && raw[i + 1] == b'[' {
            return Some(i + 2);
        }
    }
    None
}

fn find_inner_end(raw: &[u8], from: usize) -> Option<usize> {
    for i in (from..raw.len().saturating_sub(1)).rev() {
        if raw[i] == b']' && raw[i + 1] == b']' {
            return Some(i);
        }
    }
    None
}

fn split_rows(inner: &[u8]) -> Vec<&[u8]> {
    let sep = b"],[";
    let mut rows = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i + 3 <= inner.len() {
        if &inner[i..i + 3] == sep {
            rows.push(&inner[start..i]);
            start = i + 3;
            i += 3;
        } else {
            i += 1;
        }
    }
    rows.push(&inner[start..]);
    rows
}

fn parse_row_sparse(row: &[u8], step: usize, n: usize, out: &mut Vec<f32>) {
    let n_out_row = (n + step - 1) / step;
    let mut col = 0usize;
    let mut out_col = 0usize;
    let mut val: u32 = 0;
    let mut in_val = false;

    for &b in row {
        if b == b',' {
            if in_val {
                if col % step == 0 {
                    out.push(val as f32);
                    out_col += 1;
                    if out_col >= n_out_row {
                        return;
                    }
                }
                col += 1;
                val = 0;
                in_val = false;
            }
        } else if b.is_ascii_digit() {
            val = val * 10 + (b - b'0') as u32;
            in_val = true;
        }
    }
    if in_val && col % step == 0 {
        out.push(val as f32);
    }
}
