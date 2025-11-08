"""
Optimal skewness and norms
"""


def l2norm(f: list[int], s: float) -> float:
    """
    Norm of a skewed polynomial: norm of f(x * sqrt(s), y / sqrt(s))
    """
    # FIXME: explain
    if len(f) == 2:
        u, v = f
        return float(u * u / s + v * v * s)
    elif len(f) == 3:
        u, v, w = f
        u, w = u / s, w * s
        return float(3 * (u * u + w * w) + 2 * u * w + v * v) / 6.0
    elif len(f) == 4:
        a, b, c, d = f
        a, c, d = a / s, c * s, d * s * s
        return (
            float(5 * (a * a + d * d) + 2 * (a * c + b * d) + b * b + c * c) / s / 8.0
        )
    elif len(f) == 5:
        a0, a1, b, c1, c0 = f
        a0, a1, c1, c0 = a0 / s**2, a1 / s, c1 * s, c0 * s**2
        return float(
            35 * (a0**2 + c0**2)
            + 10 * b * (a0 + c0)
            + 5 * (a1**2 + c1**2)
            + 6 * (a0 * c0 + a1 * c1)
            + 3 * b**2
        )
    elif len(f) == 6:
        a0, a1, a2, c2, c1, c0 = f
        a0, a1 = a0 / s**2, a1 / s
        c2, c1, c0 = c2 * s, c1 * s**2, c0 * s**3
        return (
            float(
                6 * (a2 * c1 + a0 * c1 + a1 * c2 + a1 * c0)
                + 14 * (c0 * c2 + a0 * a2)
                + 63 * (a0**2 + c0**2)
                + 7 * (a1**2 + c1**2)
                + 3 * (a2**2 + c2**2)
            )
            / s
        )
    else:
        raise NotImplementedError


def skewness(f: list[int]) -> float:
    """
    Equivalent of Cado-NFS skew_l2norm_tk_circular
    """
    # Coefficients are less than 10**50, this should fit double-precision range
    # The norm is assumed to be a decreasing-then-increasing function of skew
    s1 = 0.1
    s2 = 1e10
    n1 = l2norm(f, s1)
    n2 = l2norm(f, s2)
    while s2 - s1 > 1e-3:
        t1 = (2 * s1 + s2) / 3.0
        t2 = (s1 + 2 * s2) / 3.0
        m1 = l2norm(f, t1)
        m2 = l2norm(f, t2)
        if m1 < m2:
            s2, n2 = t2, m2
        else:
            s1, n1 = t1, m1
    if n1 < n2:
        return s1
    else:
        return s2
