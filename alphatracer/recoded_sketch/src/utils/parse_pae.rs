use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};

/// Parse a single PAE JSON file, subsampling every `step`-th row and column.
/// Returns (n_full, n_out, raw_f32_bytes) where raw bytes are little-endian f32.
pub fn parse_file(path: &str, step: usize) -> Result<(u32, u32, Vec<u8>), String> {
    let raw = fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    let inner_start = find_inner_start(&raw).ok_or_else(|| format!("No [[ in {path}"))?;
    let inner_end   = find_inner_end(&raw, inner_start).ok_or_else(|| format!("No ]] in {path}"))?;

    let rows  = split_rows(&raw[inner_start..inner_end]);
    let n     = rows.len();
    let n_out = (n + step - 1) / step;

    let mut values: Vec<f32> = Vec::with_capacity(n_out * n_out);
    for (row_i, row_bytes) in rows.iter().enumerate() {
        if row_i % step == 0 {
            parse_row_sparse(row_bytes, step, n, &mut values);
        }
    }

    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((n as u32, n_out as u32, raw_bytes))
}

#[allow(dead_code)]
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

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for path in &args[2..] {
        match parse_file(path, step) {
            Ok((n_full, n_out, raw_bytes)) => {
                out.write_all(&n_full.to_le_bytes()).unwrap();
                out.write_all(&n_out.to_le_bytes()).unwrap();
                out.write_all(&raw_bytes).unwrap();
            }
            Err(e) => { eprintln!("{e}"); std::process::exit(1); }
        }
    }
}

fn find_inner_start(raw: &[u8]) -> Option<usize> {
    for i in 0..raw.len().saturating_sub(1) {
        if raw[i] == b'[' && raw[i + 1] == b'[' { return Some(i + 2); }
    }
    None
}

fn find_inner_end(raw: &[u8], from: usize) -> Option<usize> {
    for i in (from..raw.len().saturating_sub(1)).rev() {
        if raw[i] == b']' && raw[i + 1] == b']' { return Some(i); }
    }
    None
}

fn split_rows(inner: &[u8]) -> Vec<&[u8]> {
    let sep = b"],[";
    let mut rows  = Vec::new();
    let mut start = 0usize;
    let mut i     = 0usize;
    while i + 3 <= inner.len() {
        if &inner[i..i + 3] == sep { rows.push(&inner[start..i]); start = i + 3; i += 3; }
        else { i += 1; }
    }
    rows.push(&inner[start..]);
    rows
}

fn parse_row_sparse(row: &[u8], step: usize, n: usize, out: &mut Vec<f32>) {
    let n_out_row = (n + step - 1) / step;
    let mut col    = 0usize;
    let mut out_col = 0usize;
    let mut val    = 0u32;
    let mut in_val = false;

    for &b in row {
        if b == b',' {
            if in_val {
                if col % step == 0 {
                    out.push(val as f32);
                    out_col += 1;
                    if out_col >= n_out_row { return; }
                }
                col += 1; val = 0; in_val = false;
            }
        } else if b.is_ascii_digit() {
            val = val * 10 + (b - b'0') as u32;
            in_val = true;
        }
        // '.' skipped — PAE values are integers
    }
    if in_val && col % step == 0 { out.push(val as f32); }
}
