// Integer division is fast on some platforms like AMD Zen 4.
const USE_EUCLID: bool = true;

/// Compute the Legendre symbol (x|p) where p is a positive odd integer.
pub(crate) fn legendre_symbol(x: i64, p: i64) -> i32 {
    // See https://en.wikipedia.org/wiki/Jacobi_symbol
    assert!(p > 0 && p & 1 == 1);
    let mut sq = 0; // Result is (-1)^sq
    let mut x = x;
    let mut n = p;
    if x < 0 {
        if n & 3 == 3 {
            sq += 1;
        }
        x = -x;
    }
    if x == 0 {
        return 0;
    }
    if p == 1 {
        return 1;
    }
    assert!(x >= 0);
    loop {
        let tz = x.trailing_zeros();
        // (2|n) = -1 iff n%8 is 3 or 5
        sq += tz & ((n & 7).count_ones() == 2) as u32;
        x >>= tz;
        // Invariant: x and n are odd
        if x == 1 {
            return if sq & 1 == 0 { 1 } else { -1 };
        }
        // Invariant: x > 1 and n > 1
        if USE_EUCLID {
            // Euclid algorithm
            if x & n & 3 == 3 {
                // Apply quadratic reciprocity
                sq += 1;
            }
            (x, n) = (n % x, x);
        } else {
            // Binary GCD style
            if x < n {
                // Apply quadratic reciprocity
                if x & n & 3 == 3 {
                    sq += 1;
                }
                (x, n) = (n - x, x);
            } else {
                (x, n) = (x - n, n);
            }
        }
        if x == 0 {
            return 0;
        }
    }
}
