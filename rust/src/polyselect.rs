use num_integer::Integer;
use std::collections::HashSet;

/// Perform the sieve step of Kleinjung's polynomial selection
/// Find i such that i mod p² = r for a given list of (p,r)
#[allow(unused)]
pub(crate) fn sieve_squares_simple(roots: &[(u64, u64)], bound: u64) -> Vec<i64> {
    let mut s = Vec::with_capacity(roots.len());
    for &(p, r0) in roots {
        let mut r = r0 as i64;
        let p2 = (p * p) as i64;
        while r < bound as i64 {
            s.push(r);
            r += p2;
        }
        r = r0 as i64 - p2 as i64;
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

pub(crate) fn sieve_squares(rootsq: &[(u64, u64)], roots: &[(u64, u64)], bound: u64) -> Vec<i64> {
    let mut s = Vec::with_capacity(rootsq.len() * roots.len());
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
            let mut r = r0;
            while r < bound as i64 {
                s.push(r);
                r += p2q2;
            }
            r = r0 - p2q2;
            while r > -(bound as i64) {
                s.push(r);
                r -= p2q2;
            }
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

/// Slow variant using a hash-set.
#[allow(unused)]
pub(crate) fn sieve_squares_hash(roots: &[(u64, u64)], bound: u64) -> Vec<i64> {
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
        r = r0 as i64 - p2 as i64;
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
