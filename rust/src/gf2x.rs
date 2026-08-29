//! Arithmetic for polynomials over GF(2) using clmul.
//!
//! All polynomials are represented by BigUint instances.

use num_bigint::BigUint;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;

/// Multiply 2 polynomial matrices (m,k) x (k,n) in row-major order
pub(crate) fn matmul(x: &[Vec<BigUint>], y: &[Vec<BigUint>]) -> Vec<Vec<BigUint>> {
    assert!(x[..].iter().all(|row| row.len() == y.len()));
    assert!(y[..].iter().all(|row| row.len() == y[0].len()));
    let k = y.len();
    let n = y[0].len();
    //println!("matmul shape {} {k} {n}", x.len());
    x.into_par_iter()
        .map(|row| {
            (0..n)
                .map(|j| {
                    let mut zij = BigUint::ZERO;
                    for idx in 0..k {
                        zij ^= mul(&row[idx], &y[idx][j]);
                    }
                    zij
                })
                .collect()
        })
        .collect()
}

pub(crate) fn mul(x: &BigUint, y: &BigUint) -> BigUint {
    mul_karatsuba(x, y)
}

pub(crate) fn mul_basic(x: &BigUint, y: &BigUint) -> BigUint {
    let xx = x.to_u64_digits();
    let yy = y.to_u64_digits();
    let mut zz = vec![0_u64; xx.len() + yy.len()];
    mul_basic_impl(&mut zz, &xx, &yy);
    BigUint::new(
        zz.into_iter()
            .flat_map(|zi| [zi as u32, (zi >> 32) as u32])
            .collect(),
    )
}

pub(crate) fn mul_karatsuba(x: &BigUint, y: &BigUint) -> BigUint {
    let xx = x.to_u64_digits();
    let yy = y.to_u64_digits();
    let mut zz = vec![0_u64; 3 * std::cmp::max(xx.len(), yy.len())];
    mul_karatsuba_impl(&mut zz, &xx, &yy);
    BigUint::new(
        zz.into_iter()
            .flat_map(|zi| [zi as u32, (zi >> 32) as u32])
            .collect(),
    )
}

// Ordinary quadratic multiplication.
fn mul_basic_impl(z: &mut [u64], x: &[u64], y: &[u64]) {
    debug_assert!(z.len() >= x.len() + y.len());
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            let zij = mul64(xi, yj);
            unsafe {
                *z.get_unchecked_mut(i + j) ^= zij[0];
                *z.get_unchecked_mut(i + j + 1) ^= zij[1];
            }
        }
    }
}

const KARATSUBA_THRESHOLD: usize = 64;

fn mul_karatsuba_impl(z: &mut [u64], x: &[u64], y: &[u64]) {
    debug_assert!(z.len() >= x.len() + y.len());
    // Make y smaller.
    let (x, y) = if x.len() < y.len() { (y, x) } else { (x, y) };
    if y.len() <= KARATSUBA_THRESHOLD {
        mul_basic_impl(z, x, y);
        return;
    }
    let half = (x.len() + 1) / 2;
    debug_assert!(x.len() <= 2 * half);
    debug_assert!(y.len() <= 2 * half);
    if y.len() <= half {
        // y is too small
        mul_karatsuba_impl(z, &x[..half], y);
        mul_karatsuba_impl(&mut z[half..], &x[half..], y);
    } else {
        // (x0 + B x1) * (y0 + B y1)
        // = (x0 * y0)(1 + B) + (x1 * y1)(B + B²) + B(x0 + x1)(y0 + y1)
        assert!(4 * half <= z.len());
        let mut tmp = vec![0_u64; 3 * half];
        mul_karatsuba_impl(&mut tmp, &x[..half], &y[..half]);
        for (i, ti) in tmp[0..2 * half].iter().enumerate() {
            unsafe {
                *z.get_unchecked_mut(i) ^= ti;
                *z.get_unchecked_mut(half + i) ^= ti;
            }
        }
        tmp.fill(0);
        mul_karatsuba_impl(&mut tmp, &x[half..], &y[half..]);
        for (i, ti) in tmp[0..2 * half].iter().enumerate() {
            unsafe {
                *z.get_unchecked_mut(half + i) ^= ti;
                *z.get_unchecked_mut(2 * half + i) ^= ti;
            }
        }

        for i in 0..half {
            tmp[i] = x[i];
            if half + i < x.len() {
                tmp[i] ^= x[half + i];
            }
            tmp[half + i] = y[i];
            if half + i < y.len() {
                tmp[half + i] ^= y[half + i];
            }
        }
        mul_karatsuba_impl(&mut z[half..], &tmp[..half], &tmp[half..2 * half]);
    }
}

#[inline(always)]
fn mul64(x: u64, y: u64) -> [u64; 2] {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("pclmulqdq") {
        unsafe { return cpu_mul64(x, y) }
    }
    mul64_doubleadd(x, y)
}

fn mul64_doubleadd(x: u64, y: u64) -> [u64; 2] {
    let mut z = 0;
    let mut y = y;
    let mut xx = x as u128;
    while y != 0 {
        let tz = y.trailing_zeros();
        y >>= tz;
        xx <<= tz;
        // Now y&1 == 1
        z ^= xx;
        y ^= 1;
    }
    [z as u64, (z >> 64) as u64]
}

fn mul64_naive(x: u64, y: u64) -> u128 {
    let mut z = 0_u128;
    for i in 0..64 {
        if (y >> i) & 1 == 1 {
            z ^= (x as u128) << i;
        }
    }
    z
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
unsafe fn cpu_mul64(x: u64, y: u64) -> [u64; 2] {
    let xx = std::arch::x86_64::_mm_set_epi64x(0, x as i64);
    let yy = std::arch::x86_64::_mm_set_epi64x(0, y as i64);
    let z = std::arch::x86_64::_mm_clmulepi64_si128(xx, yy, 0);
    let z0 = std::arch::x86_64::_mm_extract_epi64(z, 0) as u64;
    let z1 = std::arch::x86_64::_mm_extract_epi64(z, 1) as u64;
    [z0, z1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul64() {
        assert_eq!(mul64_naive(57, 73), 4017);
        assert_eq!(mul64_doubleadd(57, 73), [4017, 0]);

        assert_eq!(mul64_naive(123456789, 1234), 135009063098);
        assert_eq!(mul64_doubleadd(123456789, 1234), [135009063098, 0]);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("pclmulqdq") {
            unsafe {
                println!("Test PCLMULQDQ");
                assert_eq!(cpu_mul64(57, 73), [4017, 0]);
                assert_eq!(cpu_mul64(123456789, 1234), [135009063098, 0]);
            }
        }
    }

    #[test]
    fn test_mul() {
        // 3^100, 5^100
        let x =
            BigUint::parse_bytes(b"515377520732011331036461129765621272702107522001", 10).unwrap();
        let y = BigUint::parse_bytes(
            b"7888609052210118054117285652827862296732064351090230047702789306640625",
            10,
        )
        .unwrap();
        let z = BigUint::parse_bytes(b"3164069441095033058739865988816956707298250006234585573456192492462248350279823648067387250025158904739380877457104929", 10).unwrap();

        assert_eq!(mul_basic(&x, &y), z);

        let xbig = x.pow(10);
        let ybig = y.pow(10);
        assert_eq!(mul_karatsuba(&xbig, &ybig), mul_basic(&xbig, &ybig));

        let xbig = x.pow(20);
        let ybig = y.pow(20);
        assert_eq!(mul_karatsuba(&xbig, &ybig), mul_basic(&xbig, &ybig));

        let xbig = x.pow(50);
        let ybig = y.pow(50);
        assert_eq!(mul_karatsuba(&xbig, &ybig), mul_basic(&xbig, &ybig));

        let xbig = x.pow(100);
        let ybig = y.pow(100);
        assert_eq!(mul_karatsuba(&xbig, &ybig), mul_basic(&xbig, &ybig));
    }
}
