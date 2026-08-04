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

/// Perform the sieve step of Kleinjung's polynomial selection
/// Find i such that i mod p² = r for a given list of (p,r) and (q,r)
/// Also return the matching primes (q, p1, p2) for each i.
pub(crate) fn sieve_squares(
    rootsq: &[(u64, u64)],
    roots: &[(u64, u64)],
    bound: u64,
) -> Vec<(i64, Vec<u64>)> {
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
}
