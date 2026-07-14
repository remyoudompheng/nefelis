use anyhow::Result;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod filter;

#[pymodule]
fn nefelis_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prune_relations, m)?)?;
    Ok(())
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
