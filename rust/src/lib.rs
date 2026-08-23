#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use num_bigint::BigInt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

mod filter;
mod math;
mod polyselect;

#[pymodule]
mod nefelis_rust {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::compute_characters;
    #[pymodule_export]
    use super::legendre_symbol;
    #[pymodule_export]
    use super::merge_relations;
    #[pymodule_export]
    use super::prune_relations;
    #[pymodule_export]
    use super::root_sieve;
    #[pymodule_export]
    use super::sieve_squares;

    #[pymodule]
    mod polys {
        use pyo3::prelude::*;

        #[pymodule_export]
        use crate::alpha;

        #[pyfunction]
        fn discriminant(f: Vec<num_bigint::BigInt>) -> num_bigint::BigInt {
            crate::polyselect::discriminant(&f)
        }

        #[pyfunction]
        fn l2norm(f: Vec<f64>) -> f64 {
            crate::polyselect::skew_l2norm(&f, 1.0)
        }

        #[pymodule_export]
        use super::super::murphy;
    }

    #[pymodule]
    mod skewpoly {
        use pyo3::prelude::*;

        #[pyfunction]
        fn l2norm(f: Vec<f64>, s: f64) -> f64 {
            crate::polyselect::skew_l2norm(&f, s)
        }

        #[pyfunction]
        fn skewness(py: Python<'_>, f: Vec<f64>) -> f64 {
            py.detach(|| crate::polyselect::skewness(&f))
        }
    }
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
    logger: Option<Py<PyAny>>,
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

#[pyfunction]
fn sieve_squares(
    py: Python<'_>,
    rootsq: Vec<(u64, u64)>,
    roots: Vec<(u64, u64)>,
    bound: u64,
) -> Vec<(i64, Vec<u64>)> {
    py.detach(|| polyselect::sieve_squares(&rootsq, &roots, bound))
}

#[pyfunction]
fn root_sieve(py: Python<'_>, f: Vec<BigInt>, g: Vec<BigInt>, bound: u64) -> PyResult<Vec<i32>> {
    if bound >= 16 << 20 {
        return Err(PyValueError::new_err(format!("Bound {bound} is too large")));
    }
    if g.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Polynomial g has degree {} instead of 1",
            g.len() as i32 - 1
        )));
    }
    let bound = bound as i32;
    let g: &[BigInt; 2] = g[..].try_into().unwrap();
    py.detach(|| match f.len() {
        3 => Ok(polyselect::root_sieve::<3>(
            bound,
            f[..].try_into().unwrap(),
            g,
        )),
        4 => Ok(polyselect::root_sieve::<4>(
            bound,
            f[..].try_into().unwrap(),
            g,
        )),
        5 => Ok(polyselect::root_sieve::<5>(
            bound,
            f[..].try_into().unwrap(),
            g,
        )),
        6 => Ok(polyselect::root_sieve::<6>(
            bound,
            f[..].try_into().unwrap(),
            g,
        )),
        lf => Err(PyValueError::new_err(format!(
            "Polynomial f have unsupported degree {}",
            lf as i64 - 1,
        ))),
    })
}

#[pyfunction]
fn alpha(py: Python<'_>, _disc: BigInt, poly: Vec<BigInt>) -> PyResult<f64> {
    py.detach(|| match poly.len() {
        3 => Ok(polyselect::alpha::<3>(poly[..].try_into().unwrap())),
        4 => Ok(polyselect::alpha::<4>(poly[..].try_into().unwrap())),
        5 => Ok(polyselect::alpha::<5>(poly[..].try_into().unwrap())),
        6 => Ok(polyselect::alpha::<6>(poly[..].try_into().unwrap())),
        l => Err(PyValueError::new_err(format!(
            "Polynomial has unsupported degree {}",
            l as i64 - 1
        ))),
    })
}

#[pyfunction]
#[rustfmt::skip]
fn murphy(
    f: Vec<f64>, g: Vec<f64>,
    alpha_f: f64, alpha_g: f64,
    area: f64,
    bf: f64, bg: f64,
    skew: f64,
) -> PyResult<f64> {
    use crate::polyselect::murphy;

    match (f.len(), g.len()) {
        // degree(g) = 1
        (5, 2) => Ok(murphy::<5, 2>(
            f[..].try_into().unwrap(), g[..].try_into().unwrap(),
            alpha_f, alpha_g, area, bf, bg, skew,
        )),
        (6, 2) => Ok(murphy::<6, 2>(
            f[..].try_into().unwrap(), g[..].try_into().unwrap(),
            alpha_f, alpha_g, area, bf, bg, skew,
        )),
        // Joux-Lercier
        (3, 2) => Ok(murphy::<3, 2>(
            f[..].try_into().unwrap(), g[..].try_into().unwrap(),
            alpha_f, alpha_g, area, bf, bg, skew,
        )),
        (4, 3) => Ok(murphy::<4, 3>(
            f[..].try_into().unwrap(), g[..].try_into().unwrap(),
            alpha_f, alpha_g, area, bf, bg, skew,
        )),
        (lf, lg) => Err(PyValueError::new_err(format!(
            "Polynomials have unsupported degrees {},{}",
            lf as i64 - 1,
            lg as i64 - 1,
        ))),
    }
}
