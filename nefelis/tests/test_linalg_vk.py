from contextlib import contextmanager
import logging

from nefelis.backends.kompute import linalg


@contextmanager
def disable_sortrows():
    linalg.DEBUG_NO_SORT_ROWS = True
    yield
    linalg.DEBUG_NO_SORT_ROWS = False


# A random large prime
ELL = 36772791633488289516136845429722005404404528067031938379713882710217889690723
# A random large integer
X = 1791299916721206690358645962640029560176392799075433219707194671759118856241


def test_linalg_small():
    rel = {"x": 1}
    rel2 = {"a": -1}
    rel3 = {"x": -2, "a": 5}
    with disable_sortrows():
        M = linalg.SpMV([rel, rel, rel2, rel3], ELL)
        print("Basis", M.basis, "dimension", M.dim)
        v = [
            pow(X, 12, ELL),
            pow(X, 24, ELL),
            pow(X, 36, ELL),
            pow(X, 48, ELL),
        ]
        w = M.polyeval(v, ELL, [0, 1])
        ix, ia = M.basis.index("x"), M.basis.index("a")
    assert w[0] == v[ix]
    assert w[1] == v[ix]
    assert w[2] == ELL - v[ia]
    assert w[3] == (5 * v[ia] - 2 * v[ix]) % ELL


def test_linalg_big():
    rel = {"SM": X}
    rel2 = {"a": -1}
    rel3 = {"SM": X + 1, "x": -2, "a": 5}

    with disable_sortrows():
        M = linalg.SpMV([rel, rel, rel2, rel3], ELL)
        print("Basis", M.basis, "dimension", M.dim)
        v = [
            pow(X, 12, ELL),
            pow(X, 24, ELL),
            pow(X, 36, ELL),
            pow(X, 48, ELL),
        ]
        w = M.polyeval(v, ELL, [0, 1])
        ism, ix, ia = [M.basis.index(k) for k in ("SM", "x", "a")]

    assert w[0] == rel["SM"] * v[ism] % ELL
    assert w[1] == rel["SM"] * v[ism] % ELL
    assert w[2] == (-v[ia]) % ELL
    assert w[3] == (rel3["SM"] * v[ism] + 5 * v[ia] - 2 * v[ix]) % ELL


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    test_linalg_small()
    test_linalg_big()
