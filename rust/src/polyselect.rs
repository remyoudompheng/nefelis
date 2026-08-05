use std::collections::HashSet;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::ops::euclid::Euclid;

/// Perform the sieve step of Kleinjung's polynomial selection
/// Find i such that i mod p² = r for a given list of (p,r)
#[allow(unused)]
fn sieve_squares_simple(roots: &[(u64, u64)], bound: u64) -> Vec<i64> {
    let mut s = Vec::with_capacity(roots.len());
    for &(p, r0) in roots {
        let mut r = r0 as i64;
        let p2 = (p * p) as i64;
        while r < bound as i64 {
            s.push(r);
            r += p2;
        }
        r = r0 as i64 - p2;
        while r > -(bound as i64) {
            s.push(r);
            r -= p2;
        }
    }
    s.sort_unstable();
    let mut res = vec![];
    for i in 1..s.len() {
        if s[i - 1] == s[i] {
            res.push(s[i]);
        }
    }
    res
}

/// Perform the sieve step of Kleinjung's polynomial selection
/// Find i such that i mod p² = r for a given list of (p,r) and (q,r)
/// Also return the matching primes (q, p1, p2) for each i.
pub(crate) fn sieve_squares(
    rootsq: &[(u64, u64)],
    roots: &[(u64, u64)],
    bound: u64,
) -> Vec<(i64, Vec<u64>)> {
    let mut v = sieve_squares_impl(rootsq, roots, bound, true);
    v.extend(sieve_squares_impl(rootsq, roots, bound, false));
    v
}

pub(crate) fn sieve_squares_impl(
    rootsq: &[(u64, u64)],
    roots: &[(u64, u64)],
    bound: u64,
    positive: bool,
) -> Vec<(i64, Vec<u64>)> {
    // Estimate array size to reserve array and avoid reallocations.
    let mut qratio = 0.0;
    let mut pratio = 0.0;
    for &(q, _) in rootsq {
        qratio += 1.0 / (q * q) as f64;
    }
    for &(p, _) in roots {
        pratio += 1.0 / (p * p) as f64;
    }
    let estimate = 1.1 * bound as f64 * pratio * qratio;
    assert!(estimate < 128e6);

    let mut s = Vec::with_capacity(estimate.round() as usize);
    for &(q, qr) in rootsq {
        let q2 = (q * q) as i64;
        for &(p, r0) in roots {
            debug_assert!(p != q);
            let p2 = (p * p) as i64;
            let p2q2 = p2.checked_mul(q2).unwrap();
            // CRT lift modulo p2 q2:
            // r = r0 + p2 (qr - r0) (p2^-1 mod q2)
            let p2inv = p2.extended_gcd(&q2).x;
            let k = p2inv
                .checked_mul(qr as i64 - r0 as i64)
                .unwrap()
                .rem_euclid(q2);
            let r0 = r0 as i64 + k * p2; // k*p2 < p2q2 cannot overflow
            debug_assert_eq!(r0 % q2, qr as i64);
            if positive {
                let mut r = r0;
                while r < bound as i64 {
                    s.push(r);
                    r += p2q2;
                }
            } else {
                let mut r = r0 - p2q2;
                while r > -(bound as i64) {
                    s.push(r);
                    r -= p2q2;
                }
            }
        }
        assert!((s.len() as f64) < estimate);
    }
    s.sort_unstable();
    let mut res = vec![];
    for i in 1..s.len() {
        if s[i - 1] == s[i] && (i + 1 == s.len() || s[i] != s[i + 1]) {
            res.push(s[i]);
        }
    }
    let mut resfactors: Vec<_> = res.into_iter().map(|si| (si, vec![])).collect();
    for &(q, qr) in rootsq {
        let q2 = q * q;
        let qinv2 = inv_2adic(q2);
        for (ref s, ref mut pq) in resfactors.iter_mut() {
            if divisible(s - qr as i64, q2, qinv2) {
                pq.push(q);
            }
        }
    }
    for &(p, pr) in roots {
        let p2 = p * p;
        let pinv2 = inv_2adic(p2);
        for (ref s, ref mut pq) in resfactors.iter_mut() {
            if divisible(s - pr as i64, p2, pinv2) {
                pq.push(p);
            }
        }
    }
    resfactors
}

fn divisible(x: i64, p: u64, pinv: u64) -> bool {
    let q = (x as u64).wrapping_mul(pinv) as i64;
    if let Some(pq) = (p as i64).checked_mul(q) {
        debug_assert_eq!(pq, x);
        true
    } else {
        false
    }
}

fn inv_2adic(n: u64) -> u64 {
    debug_assert_eq!(n & 1, 1);
    let mut x = n; // nx=1 mod 8
    for _ in 0..5 {
        x = (x << 1).wrapping_sub(x.wrapping_mul(x).wrapping_mul(n));
    }
    x
}

/// Slow variant using a hash-set.
#[allow(unused)]
fn sieve_squares_hash(roots: &[(u64, u64)], bound: u64) -> Vec<i64> {
    let mut s = HashSet::<i64>::new();
    let mut res = vec![];
    for &(p, r0) in roots {
        let mut r = r0 as i64;
        let p2 = (p * p) as i64;
        while r < bound as i64 {
            if s.contains(&r) {
                res.push(r);
            } else {
                s.insert(r);
            }
            r += p2;
        }
        r = r0 as i64 - p2;
        while r > -(bound as i64) {
            if s.contains(&r) {
                res.push(r);
            } else {
                s.insert(r);
            }
            r -= p2;
        }
    }
    res
}

/// Root sieve: find λ such that f+λg has maximal score (roots modulo small primes).
/// Returns a list of candidates.
pub(crate) fn root_sieve<const D: usize>(bound: i32, f: &[BigInt; D], g: &[BigInt; 2]) -> Vec<i32> {
    assert!(bound < 16 << 20);
    let mut sieve = vec![0.0_f32; 2 * bound as usize];
    let mut buf = [0.0_f32; 128];
    let mut discs = vec![]; // precompute discriminants
    let mut fplus = f.clone();
    for _ in 0..buf.len() {
        discs.push(discriminant(&fplus));
        fplus[0] += &g[0];
        fplus[0] += &g[1];
    }
    for (lidx, &l) in SMALLPRIMES.iter().enumerate() {
        let lbig = BigInt::from(l);
        let mut fl: [i32; D] = std::array::from_fn(|i| f[i].rem_euclid(&lbig).try_into().unwrap());
        let gl: [i32; 2] = std::array::from_fn(|i| g[i].rem_euclid(&lbig).try_into().unwrap());
        let lf = l as f32;
        let factor = lf.log2() * lf / (lf + 1.);
        let stride = match l {
            2 => 64,
            3 => 81,
            5 => 125,
            7 | 11 => l * l,
            _ => l,
        };
        for i in 0..stride as usize {
            let v = avgval(lidx, &fl, &discs[i] % l == BigInt::ZERO);
            buf[i] = v as f32 * factor;
            fl[0] += gl[0];
            if fl[0] >= l {
                fl[0] -= l
            }
            fl[1] += gl[1];
            if fl[1] >= l {
                fl[1] -= l
            }
        }
        // Fill array
        let start = bound.rem_euclid(stride); // -bound+start is multiple of l
        for i in 0..start {
            sieve[i as usize] += buf[(stride - start + i) as usize];
        }
        let mut i = start;
        while ((i + stride) as usize) < sieve.len() {
            for j in 0..stride as usize {
                sieve[i as usize + j] += buf[j];
            }
            i += stride;
        }
        for j in 0..(sieve.len() - (i as usize)) {
            sieve[i as usize + j] += buf[j];
        }
    }
    // Now select top-k elements.
    let smax = sieve.iter().copied().reduce(f32::max).unwrap();
    sieve
        .into_iter()
        .enumerate()
        .filter_map(|(idx, s)| {
            if s > smax - 0.5 {
                Some(idx as i32 - bound)
            } else {
                None
            }
        })
        .collect()
}

// Polynomial statistics and valuations.

static SMALLPRIMES: [i32; 30] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113,
];

type FpPowers = [Vec<i32>; 4]; // arrays of x, x², x³, x^4 mod l
static FIELDS: std::sync::LazyLock<[FpPowers; 30]> = std::sync::LazyLock::new(|| {
    SMALLPRIMES.map(|l| {
        let mut pows = std::array::from_fn(|_| vec![0i32; l as usize]);
        for x in 0..l {
            let mut xy = x;
            for y in 0..pows.len() {
                pows[y][x as usize] = xy;
                xy = (xy * x) % l;
            }
        }
        pows
    })
});

/// Compute α(poly) in base 2 (log2 instead of log).
pub(crate) fn alpha<const D: usize>(poly: &[BigInt; D]) -> f64 {
    let mut a = 0.0;
    let disc = discriminant(poly);
    for (lidx, &l) in SMALLPRIMES.iter().enumerate() {
        let lbig = if l == 2 {
            BigInt::from(32)
        } else {
            BigInt::from(l.pow(2))
        };
        let mut f: [i32; D] =
            std::array::from_fn(|i| poly[i].rem_euclid(&lbig).try_into().unwrap());
        if f[D - 1] % l == 0 {
            f.reverse();
        }
        let al = avgval(lidx, &f, &disc % l == BigInt::ZERO);
        let lf = l as f64;
        let alpha_l = lf.log2() * (1. / (lf - 1.) - al * lf / (lf + 1.));
        //println!("{l} {alpha_l} {al}");
        a += alpha_l;
    }
    a
}

fn avgval<const D: usize>(lidx: usize, poly: &[i32; D], ramified: bool) -> f64 {
    debug_assert!(D <= 5);
    let l = SMALLPRIMES[lidx];
    let pows = &FIELDS[lidx];
    if !ramified {
        let n = nroots(pows, l, poly);
        return n as f64 / (l - 1) as f64;
    }
    // Otherwise, count roots modulo a small power of l.
    if l == 2 {
        // Count roots modulo 32
        let mut res = 0.0;
        for x in 0..32 {
            let mut fx = poly[D - 1];
            for j in 2..=D {
                fx = fx.wrapping_mul(x).wrapping_add(poly[D - j]);
            }
            res += std::cmp::min(5, fx.trailing_zeros()) as f64 / 32.0;
        }
        // And roots at infinity
        if poly[D - 1] & 1 == 0 {
            for x in 0..16 {
                let mut fx = poly[0];
                for j in 1..D {
                    fx = fx.wrapping_mul(x).wrapping_add(poly[j]);
                }
                res += std::cmp::min(5, fx.trailing_zeros()) as f64 / 32.0;
            }
        }
        return res;
    }
    let lf = l as f64;
    let mut res = 0.0;
    let l2 = (l * l) as i64;
    let r1 = 1. / lf;
    let r2 = 1. / (lf * (lf - 1.));
    for x in 0..l as usize {
        let mut v = poly[0];
        unsafe {
            for i in 1..D {
                v += poly[i] * pows.get_unchecked(i - 1).get_unchecked(x);
            }
        }
        v %= l;
        if v == 0 {
            res += r1;
            let mut x2 = x as i64;
            for _ in 0..l {
                let mut fx = poly[D - 1] as i64;
                for j in 2..=D {
                    fx = fx * x2 + poly[D - j] as i64;
                    if j == 3 {
                        fx %= l2;
                    }
                }
                fx %= l2;
                if fx == 0 {
                    res += r2;
                }
                x2 += l as i64;
            }
        }
    }
    // Add roots at infinity
    if poly[D - 1] % l == 0 {
        // revf(lx) = revf[0] + revf[1] * lx + O(l^2)
        res += r1;
        for j in 0..l {
            let fx = (poly[D - 1] + poly[D - 2] * j * l) as i64;
            if fx % l2 == 0 {
                res += r2;
            }
        }
    }
    res
}

fn nroots<const D: usize>(pows: &FpPowers, l: i32, poly: &[i32; D]) -> usize {
    debug_assert_eq!(pows[0].len(), l as usize);
    debug_assert!(D <= 5);
    let mut res = usize::from(poly[D - 1] % l == 0); // root at infinity
    for x in 0..l as usize {
        let mut v = poly[0];
        unsafe {
            for i in 1..D {
                v += poly[i] * pows.get_unchecked(i - 1).get_unchecked(x);
            }
        }
        v %= l;
        if v == 0 {
            res += 1;
        }
    }
    res
}

pub(crate) fn discriminant(f: &[BigInt]) -> BigInt {
    match f.len() {
        3 => &f[1] * &f[1] - 4 * &f[0] * &f[2],
        4 => {
            let [d, c, b, a] = f else { unreachable!() };
            let (ac, bd, bb, cc, ad) = (a * c, b * d, b * b, c * c, a * d);
            18 * &ac * &bd + &bb * &cc - 4 * bb * bd - 4 * ac * cc - 27 * &ad * &ad
        }
        5 => {
            let [e, d, c, b, a] = f else { unreachable!() };
            let disc0 = c * c - 3 * b * d + 12 * a * e;
            let disc1 =
                2 * c.pow(3) - 9 * b * c * d + 27 * b * b * e + 27 * a * d * d - 72 * a * c * e;
            (4 * &disc0 * &disc0 * &disc0 - &disc1 * &disc1) / 27
        }
        _ => panic!("unsupported"),
    }
}

/// Norm of a skewed polynomial: norm of f(x * sqrt(s), y / sqrt(s))
pub(crate) fn skew_l2norm(f: &[f64], s: f64) -> f64 {
    // FIXME: explain
    match f.len() {
        2 => {
            let u = f[0];
            let v = f[1];
            u * u / s + v * v * s
        }
        3 => {
            let u = f[0] / s;
            let v = f[1];
            let w = f[2] * s;
            (3.0 * (u * u + w * w) + 2.0 * u * w + v * v) / 6.0
        }
        4 => {
            let a = f[0] / s;
            let b = f[1];
            let c = f[2] * s;
            let d = f[3] * s * s;
            (5.0 * (a * a + d * d) + 2.0 * (a * c + b * d) + b * b + c * c) / s / 8.0
        }
        5 => {
            let a0 = f[0] / s.powi(2);
            let a1 = f[1] / s;
            let b = f[2];
            let c1 = f[3] * s;
            let c0 = f[4] * s.powi(2);
            35.0 * (a0.powi(2) + c0.powi(2))
                + 10.0 * b * (a0 + c0)
                + 5.0 * (a1.powi(2) + c1.powi(2))
                + 6.0 * (a0 * c0 + a1 * c1)
                + 3.0 * b.powi(2)
        }
        6 => {
            let a0 = f[0] / s.powi(2);
            let a1 = f[1] / s;
            let a2 = f[2];
            let c2 = f[3] * s;
            let c1 = f[4] * s.powi(2);
            let c0 = f[5] * s.powi(3);
            (6.0 * (a2 * c1 + a0 * c1 + a1 * c2 + a1 * c0)
                + 14.0 * (c0 * c2 + a0 * a2)
                + 63.0 * (a0.powi(2) + c0.powi(2))
                + 7.0 * (a1.powi(2) + c1.powi(2))
                + 3.0 * (a2.powi(2) + c2.powi(2)))
                / s
        }
        _ => panic!("unsupported degree"),
    }
}

/// Equivalent of Cado-NFS skew_l2norm_tk_circular
pub(crate) fn skewness(f: &[f64]) -> f64 {
    // Coefficients are less than 10**50, this should fit double-precision range
    // The norm is assumed to be a decreasing-then-increasing function of skew
    let mut s1 = 0.1;
    let mut s2 = 1e10;
    let mut n1 = skew_l2norm(&f, s1);
    let mut n2 = skew_l2norm(&f, s2);
    while s2 - s1 > 1e-3 {
        let t1 = (2.0 * s1 + s2) / 3.0;
        let t2 = (s1 + 2.0 * s2) / 3.0;
        let m1 = skew_l2norm(&f, t1);
        let m2 = skew_l2norm(&f, t2);
        if m1 < m2 {
            s2 = t2;
            n2 = m2;
        } else {
            s1 = t1;
            n1 = m1;
        }
    }
    if n1 < n2 {
        s1
    } else {
        s2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_2adic() {
        assert_eq!(inv_2adic(43), 9437869060967677571);
        assert_eq!(inv_2adic(1337), 2938785705086114057);
        assert_eq!(inv_2adic(27902742321460541), 16549517751502637589);
        assert_eq!(inv_2adic(7511906879876805787), 9159278723288011155);
        for i in 0..1024 {
            let x = 15833507808570735471 + 2 * i;
            let y = inv_2adic(x);
            assert_eq!(x.wrapping_mul(y), 1);
        }
    }

    #[test]
    fn test_divisible() {
        assert!(divisible(867303362901100111, 1337, 2938785705086114057));
        assert!(!divisible(1141653610437229729, 1337, 2938785705086114057));
    }

    #[test]
    fn test_nroots() {
        assert_eq!(SMALLPRIMES[9], 29);
        assert_eq!(nroots(&FIELDS[9], 29, &[1, 6, 7]), 0);
        assert_eq!(nroots(&FIELDS[9], 29, &[3, 1, 7]), 2);
        assert_eq!(nroots(&FIELDS[9], 29, &[3, 5, 7, 4]), 0);
        assert_eq!(nroots(&FIELDS[9], 29, &[3, 1, 7, 4]), 1);
        assert_eq!(nroots(&FIELDS[9], 29, &[2, 5, 7, 4]), 3);
        assert_eq!(nroots(&FIELDS[9], 29, &[18, 5, 7, 4, 11]), 0);
        assert_eq!(nroots(&FIELDS[9], 29, &[6, 5, 7, 4, 11]), 1);
        assert_eq!(nroots(&FIELDS[9], 29, &[16, 5, 7, 4, 11]), 2);
        assert_eq!(nroots(&FIELDS[9], 29, &[26, 5, 7, 4, 11]), 4);
    }

    #[test]
    fn test_alpha() {
        // expect alpha = 1.91
        let a = alpha(&[2, 2, 3, 1].map(BigInt::from));
        println!("α = {a}");
        assert!(1.90 < a && a < 1.96, "{a}");

        // expect alpha = 0.862
        let a = alpha(&[5, 1, -4, 2, 3].map(BigInt::from));
        println!("α = {a}");
        assert!(0.86 < a && a < 0.89, "{a}");

        // expect alpha = -1.74
        let f: [i64; _] = [
            14210657027941395,
            89584037279263219,
            45122821949983494,
            99446805877972590,
            33124700204200920,
        ];
        let a = alpha(&f.map(BigInt::from));
        println!("α = {a}");
        assert!(-1.8 < a && a < -1.4, "{a}");
    }

    #[test]
    fn test_discriminant() {
        assert_eq!(discriminant(&[2, 3, 4].map(BigInt::from)), (-23).into());
        assert_eq!(discriminant(&[4, 3, 2].map(BigInt::from)), (-23).into());
        assert_eq!(
            discriminant(&[7, 1, 9, 4].map(BigInt::from)),
            (-36979).into()
        );
        assert_eq!(
            discriminant(&[7, 1, 9, 4, 2].map(BigInt::from)),
            1097356.into()
        );
    }

    #[test]
    fn test_skew() {
        let f: Vec<f64> = vec![
            -148542579752395458097583792765647.,
            48704989161145263711343080.,
            596511211626418813.,
            -11378139486.,
            480.,
        ];
        let s = skewness(&f);
        assert_eq!(s.round(), 29293087.);
        let n = skew_l2norm(&f, s);
        assert!(2.0103329e37 <= n && n <= 2.0103330e37);
    }
}
