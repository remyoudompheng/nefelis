#![allow(non_snake_case)]

//! Computation of linear generators for GF(2) matrix sequences
//! The implementation follows the conventions of:
//!
//! Emmanuel Thomé
//! Fast computation of linear generators for matrix sequences and application to the block Wiedemann algorithm.
//! ISSAC '01: Proceedings of the 2001 international symposium on Symbolic and algebraic computation, Jul 2001, London, Ontario, Canada. pp.323-331,
//! https://inria.hal.science/inria-00517999
//!
//! All matrices are represented in column-major order.
//! Only square sizes m=n=32 is supported in pratice.

use num_bigint::BigUint;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;

use crate::gf2x;

/// Compute a transition matrix such that (E x P)[0] = 0
/// and delta is a valid profile.
///
/// E has shape (m,m+n) represented by m integers with (m+n) bits
fn lingen_step32<const M: usize, const MN: usize>(
    E: &mut [[u32; M]; MN],
    delta: &mut [u32; MN],
    P: &mut [[u32; MN]; MN],
    iter: usize,
) {
    // Sort columns
    for i in 0..MN {
        let ii = if iter > 0 {
            // delta is almost sorted (deltaprev + {0,1} where deltaprev is sorted)
            let mut ii = i;
            for j in (i + 1)..MN {
                if delta[j] < delta[i] {
                    ii = j;
                    break;
                }
                if delta[j] > delta[i] {
                    break; // delta[jj > j] is always >= delta[i]
                }
            }
            debug_assert_eq!(ii, (i..MN).min_by_key(|&idx| delta[idx as usize]).unwrap());
            ii
        } else {
            (i..MN).min_by_key(|&idx| delta[idx as usize]).unwrap()
        };
        if i < ii {
            // Swap columns
            delta.swap(i, ii);
            P.swap(i, ii);
            E.swap(i, ii);
        }
    }
    debug_assert!((1..MN).all(|i| delta[i - 1] <= delta[i]));
    // Elimination
    let mut nonzero = [false; MN];
    for i in 0..M {
        let Some(j0) = (0..MN)
            .filter(|&j| E[j as usize][i] & 1 == 1 && !nonzero[j as usize])
            .next()
        else {
            continue;
        };
        nonzero[j0] = true;
        debug_assert!(E[j0][i] & 1 == 1);
        for j in (j0 + 1)..MN {
            if E[j][i] & 1 == 1 {
                for h in 0..M {
                    E[j][h] ^= E[j0][h];
                }
                for h in 0..MN {
                    P[j][h] ^= P[j0][h];
                }
            }
            debug_assert!(E[j][i] & 1 == 0);
        }
    }
    for j in 0..MN {
        if nonzero[j] {
            debug_assert!((0..M).any(|i| E[j][i] & 1 == 1));
            delta[j] += 1;
            for i in 0..MN {
                P[j][i] <<= 1;
            }
        } else {
            debug_assert!((0..M).all(|i| E[j][i] & 1 == 0));
            for i in 0..M {
                E[j][i] >>= 1;
            }
        }
    }
}

const MSLGDC_THRESHOLD: usize = 32;

/// Divide and conquer algorithm for MSLGDC.
pub(crate) fn mslgdc<const M: usize, const MN: usize>(
    E: &mut [[BigUint; M]; MN],
    delta: &[u32; MN],
    b: usize,
    iter: usize,
) -> Box<[[BigUint; MN]; MN]> {
    if b < MSLGDC_THRESHOLD {
        // Iterative variant
        let mut P = [[0; MN]; MN];
        for i in 0..MN {
            P[i][i] = 1;
        }
        let mut esmall: [[u32; M]; MN] =
            std::array::from_fn(|j| std::array::from_fn(|i| (&E[j][i]).try_into().unwrap()));
        let delta_before = delta.iter().copied().sum::<u32>();
        let mut delta = *delta;
        for i in 0..b {
            lingen_step32(&mut esmall, &mut delta, &mut P, i);
        }
        let delta_after = delta.iter().copied().sum::<u32>();
        assert!(delta_after - delta_before <= (M * b) as u32);
        for i in 0..M {
            for j in 0..MN {
                E[j][i] = esmall[j][i].into();
            }
        }
        return Box::new(P.map(|col| col.map(BigUint::from)));
    }
    if iter > 0 {
        // Invariant: E has degree < b
        debug_assert!(
            E[..]
                .iter()
                .all(|col| col.iter().all(|eij| eij.bits() <= b as u64)),
            "b={b} iter={iter}"
        );
    }
    // WARNING: large arrays must not be put on stack (e.g. if M=64 and MN=128)

    // Low degree size
    let mask = (BigUint::ONE << (b / 2)) - BigUint::ONE;
    let maskhi = (BigUint::ONE << b) - BigUint::ONE;
    let mut EL: Box<[[BigUint; M]; MN]> = Box::new(std::array::from_fn(|j| {
        std::array::from_fn(|i| &E[j][i] & &mask)
    }));
    let PL = mslgdc::<M, MN>(&mut EL, delta, b / 2, iter);

    // Compute (E @ PL)[b-b/2:]
    // We compute columns in parallel
    // EL is not needed anymore
    let mut ER = EL;
    let cols: Vec<[BigUint; M]> = (0..MN)
        .into_par_iter()
        .map(|j| {
            std::array::from_fn(|i| {
                let mut colij = BigUint::ZERO;
                for k in 0..MN {
                    // Eik * Pkj
                    let mij = gf2x::mul(&E[k][i], &PL[j][k]);
                    // Truncate to bits b/2..b
                    colij ^= (mij & &maskhi) >> (b / 2);
                }
                colij
            })
        })
        .collect();
    for (j, colj) in cols.into_iter().enumerate() {
        ER[j] = colj;
    }

    // Delta = Delta * PL
    let delta_r: [u32; MN] = std::array::from_fn(|j| {
        (0..MN)
            .filter(|&i| PL[j][i] != BigUint::ZERO)
            .map(|i| delta[i] + PL[j][i].bits() as u32 - 1)
            .max()
            .unwrap()
    });
    //println!("{} {:?}", iter + b / 2, delta_r);

    let PR = mslgdc::<M, MN>(&mut ER, &delta_r, b - b / 2, iter + b / 2);
    drop(ER);

    // Now compute P = PL @ PR
    // We compute columns in parallel
    let mut P: Box<[[BigUint; MN]; MN]> = Box::new(std::array::from_fn(|_| {
        std::array::from_fn(|_| BigUint::ZERO)
    }));
    let cols: Vec<[BigUint; MN]> = (0..MN)
        .into_par_iter()
        .map(|j| {
            std::array::from_fn(|i| {
                let mut colij = BigUint::ZERO;
                for k in 0..MN {
                    // PLik * PRkj
                    let mij = gf2x::mul(&PL[k][i], &PR[j][k]);
                    colij ^= mij;
                }
                // P has degree <= b
                debug_assert!(colij.bits() <= b as u64 + 1);
                colij
            })
        })
        .collect();
    for (j, colj) in cols.into_iter().enumerate() {
        P[j] = colj;
    }
    P
}
