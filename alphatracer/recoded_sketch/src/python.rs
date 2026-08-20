use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use crate::{build_alphabet, load_index, read_fasta, run_search, sketch};
use crate::parse_pae;
use crate::build_index as lib_build_index;

#[pyfunction]
#[pyo3(signature = (sidx_path, fasta_path, top_k=5, min_shared=2, n_hash_search=0))]
pub fn search_fasta(
    py: Python<'_>,
    sidx_path: String,
    fasta_path: String,
    top_k: usize,
    min_shared: u32,
    n_hash_search: usize,
) -> PyResult<Vec<(String, u32, u32, f64)>> {
    let out = py.allow_threads(move || {
        let idx    = load_index(&sidx_path);
        let (dayhoff_arr, n_letters) = build_alphabet(&idx.scheme);
        let base   = n_letters as u32;
        let k      = idx.k;
        let n_hash = idx.n_hash;
        let n_hs   = if n_hash_search == 0 { n_hash } else { n_hash_search };
        let dayhoff = Arc::new(dayhoff_arr);

        let (query_ids, query_seqs) = read_fasta(&fasta_path);
        let dh = Arc::clone(&dayhoff);
        let query_sketches: Vec<Vec<u32>> = query_seqs.par_iter()
            .map(|s| sketch(s, &dh, k, n_hash, base))
            .collect();

        let progress = AtomicUsize::new(0);
        let results  = run_search(&idx, &query_sketches, min_shared, top_k, n_hs, &progress);
        idx.advise_free();

        let mut out = Vec::new();
        for (qi, hits) in results.iter().enumerate() {
            for &(shared, seq_idx) in hits {
                out.push((query_ids[qi].clone(), seq_idx, shared, shared as f64 / n_hs as f64));
            }
        }
        out
    });
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (paths, step=4))]
pub fn parse_pae_batch(
    py: Python<'_>,
    paths: Vec<String>,
    step: usize,
) -> PyResult<Vec<(String, u32, u32, PyObject)>> {
    // Parse without GIL (parallel)
    let raw: Vec<(String, u32, u32, Vec<u8>)> = py.allow_threads(move || {
        paths.par_iter().map(|p| {
            let (n_full, n_out, raw) = parse_pae::parse_file(p, step)
                .unwrap_or_else(|e| panic!("{e}"));
            (p.clone(), n_full, n_out, raw)
        }).collect()
    });
    // Convert raw bytes to Python bytes objects (requires GIL)
    Ok(raw.into_iter().map(|(p, n_full, n_out, bytes)| {
        (p, n_full, n_out, PyBytes::new_bound(py, &bytes).into_any().unbind())
    }).collect())
}

#[pyfunction]
#[pyo3(signature = (seq_pqs, out_path, k=11usize, max_freq=0.001f64, n_hash=64usize, scheme="murphy2000_5"))]
pub fn build_index(
    py: Python<'_>,
    seq_pqs: Vec<String>,
    out_path: String,
    k: usize,
    max_freq: f64,
    n_hash: usize,
    scheme: &str,
) -> PyResult<usize> {
    let scheme = scheme.to_string();
    py.allow_threads(move || {
        lib_build_index(&seq_pqs, &out_path, k, max_freq, n_hash, &scheme)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pae_batch, m)?)?;
    m.add_function(wrap_pyfunction!(build_index, m)?)?;
    Ok(())
}
