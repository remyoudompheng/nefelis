use anyhow::Result;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

mod filter;
mod math;

#[pymodule]
fn nefelis_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prune_relations, m)?)?;
    m.add_function(wrap_pyfunction!(merge_relations, m)?)?;
    m.add_function(wrap_pyfunction!(legendre_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(compute_characters, m)?)?;
    Ok(())
}

#[pyfunction]
fn legendre_symbol(x: i64, p: i64) -> i32 {
    math::legendre_symbol(x, p)
}

#[pyfunction]
#[pyo3(signature=(filename, logger=None),
        text_signature="(filename: str, logger=None)")]
fn prune_relations(py: Python<'_>, filename: &str, logger: Option<Py<PyAny>>) -> PyResult<()> {
    prune_relations_impl(py, filename, logger)
        .map_err(|e| PyValueError::new_err(format!("Could not parse file: {e}")))?;
    Ok(())
}

fn prune_relations_impl(py: Python<'_>, filename: &str, logger: Option<Py<PyAny>>) -> Result<()> {
    let logfunc = |s: String| {
        if let Some(l) = &logger {
            let _ = l.call_method1(py, "info", (s,));
        }
    };
    filter::prune_singles(filename, &format!("{filename}.pruned.singles"), logfunc)?;
    filter::prune_cliques(
        &format!("{filename}.pruned.singles"),
        &format!("{filename}.pruned.1"),
        logfunc,
    )?;
    filter::prune_cliques(
        &format!("{filename}.pruned.1"),
        &format!("{filename}.pruned.2"),
        logfunc,
    )?;
    filter::prune_cliques(
        &format!("{filename}.pruned.2"),
        &format!("{filename}.pruned"),
        logfunc,
    )
}

#[pyfunction]
#[pyo3(signature=(filename, characters, logger=None),
        text_signature="(filename: str, characters: list[tuple[int, int]], logger=None)")]
fn merge_relations(
    py: Python<'_>,
    filename: &str,
    characters: Vec<(i64, i64)>,
    logger: Option<Py<PyAny>>
) -> PyResult<Py<PyAny>> {
    let logfunc = |s: String| {
        if let Some(l) = &logger {
            let _ = l.call_method1(py, "info", (s,));
        }
    };
    let (rels, zrels) = filter::parse_with_characters(filename, characters)
        .map_err(|e| PyValueError::new_err(format!("Could not parse file: {e}")))?;

    let filtered = filter::filter_gf2(rels, logfunc);
    filter::write_filtered(&format!("{filename}.filtered"), &filtered)
        .map_err(|e| PyValueError::new_err(format!("Could not parse file: {e}")))?;

    let filtered_packed: Vec<Vec<u8>> = filtered.into_iter().map(filter::le32_vector).collect();
    (filtered_packed, zrels).into_py_any(py)
}

#[pyfunction]
#[pyo3(signature=(xys, characters),
        text_signature="(xys: list[tuple[int, int]], characters: list[tuple[int, int]])")]
fn compute_characters(xys: Vec<(i64, i64)>, characters: Vec<(i64, i64)>) -> Vec<u64> {
    assert!(characters.len() <= 64);
    let mut res = Vec::with_capacity(xys.len());
    for (x, y) in xys {
        let mut mask = 0_u64;
        // Compute characters
        for (cidx, &(l, r)) in characters.iter().enumerate() {
            let v = x as i128 - r as i128 * y as i128;
            if legendre_symbol((v % l as i128) as i64, l) < 0 {
                mask |= 1 << cidx;
            }
        }
        res.push(mask);
    }
    res
}
