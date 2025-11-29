"""
Compatibility layer for Cado-NFS
"""


def export_polys(fd, n: int, skew: float, f: list[int], g: list[int]):
    """
    Write polynomials in a Cado-NFS compatible formatted file.
    """
    fd.write(f"n: {n}\n")
    fd.write(f"skew: {skew:.6f}\n")
    for i, fi in enumerate(f):
        fd.write(f"c{i}: {fi}\n")
    for i, gi in enumerate(g):
        fd.write(f"Y{i}: {gi}\n")
    fd.write("# MurphyE (Bf=1,Bg=1,area=1) = 0.0\n")
    fd.write(f"# f(x) = {poly_str(f)}\n")
    fd.write(f"# g(x) = {poly_str(g)}\n")


def import_polys(fd) -> tuple[int, list[int], list[int]]:
    """
    Read polynomials from a text file in Cado-NFS format.
    """
    n, f, g = None, [], []
    for line in fd:
        if line.startswith("n: "):
            n = int(line.split()[1])
        if line.startswith("c"):
            sep = line.index(":")
            idx = int(line[1:sep])
            fi = int(line.split()[1])
            while idx >= len(f):
                f.append(0)
            f[idx] = fi
        if line.startswith("Y"):
            sep = line.index(":")
            idx = int(line[1:sep])
            gi = int(line.split()[1])
            while idx >= len(g):
                g.append(0)
            g[idx] = gi

    if n is None or not f or not g:
        raise ValueError("missing fields in polynomial file")

    return n, f, g


def poly_str(f):
    """
    >>> poly_str([-1, 0, 2, 3])
    '3*x^3+2*x^2-1'
    >>> poly_str([0, 1, -1])
    '-x^2+x'
    >>> poly_str([-2, 2, 1])
    'x^2+2*x-2'
    """
    fstr = ""
    for i, fi in reversed(list(enumerate(f))):
        if fi == 0:
            continue
        coeff = f"{fi:+}"
        if i == 0:
            var = ""
        elif i == 1:
            var = "*x"
        else:
            var = f"*x^{i}"
        if i == len(f) - 1:
            coeff = coeff.lstrip("+")
        if abs(fi) == 1 and var:
            coeff = coeff.rstrip("1")
            var = var.lstrip("*")
        fstr += f"{coeff}{var}"
    return fstr
