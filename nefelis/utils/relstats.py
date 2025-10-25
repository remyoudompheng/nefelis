"""
Analyze a relation file and print statistics about largest primes.
This can be used to tune sieving bounds.

It is also tentatively compatible with Cado-NFS relation files.
"""

import argparse
import gzip
import math
import sys

import numpy
from nefelis.integers import product


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("RELFILE", help="Sieve output file")
    args = argp.parse_args()

    file = args.RELFILE
    if file.endswith(".gz"):
        with gzip.open(file, "rt") as f:
            process_file(f)
    else:
        with open(file) as f:
            process_file(f)


def process_file(f):
    sum0, sum1 = 0, 0
    count = 0
    xs, ys = [], []
    p1, p2, p3 = [], [], []
    q1, q2, q3, q4 = [], [], [], []
    q1co, q1q2co = [], []
    qB1, qB2 = None, None
    sieving = []
    side_stats = []
    side = None
    for file in sys.argv[1:]:
        with gzip.open(file, "rt") if file.endswith(".gz") else open(file) as f:
            for line in f:
                # Look for q=<integer>
                if "q=" in line:
                    qs = next(w for w in line.split() if w.startswith("q="))
                    q = int(qs[2:].strip(";:, "))
                    sieving.append(q)
                    if len(sieving) > 10:
                        sieving = sieving[-10:]
                if line.startswith("# "):
                    continue
                xy, rel0, rel1 = line.strip().split(":")
                x, y = [int(x) for x in xy.split(",")]
                xs.append(x)
                ys.append(y)
                rel0raw = [int(x, 16) for x in rel0.split(",")]
                rel1raw = [int(x, 16) for x in rel1.split(",")]
                if len(q1) < 10:
                    if any(q in rel0raw for q in sieving):
                        side_stats.append(0)
                    else:
                        side_stats.append(1)
                elif len(q1) == 10:
                    if side_stats.count(0) > side_stats.count(1):
                        print("Sieving is on side 0")
                        side = 0
                    else:
                        print("Sieving is on side 1")
                        side = 1
                rel0, rel1 = rel0raw, rel1raw
                if side == 0:
                    rel0 = [l for l in rel0raw if l not in sieving]
                elif side == 1:
                    rel1 = [l for l in rel1raw if l not in sieving]
                rel0.sort()
                rel1.sort()
                p1.append(rel0[-1])
                p2.append(rel0[-2])
                if len(rel0) > 3:
                    p3.append(rel0[-3])
                q1.append(rel1[-1])
                if len(rel1) > 1:
                    q2.append(rel1[-2])
                if len(rel1) > 2:
                    q3.append(rel1[-3])
                if len(rel1) > 3:
                    q4.append(rel1[-4])

                # Statistics of products on the opposite side
                if side == 0:
                    q1co.append(product(rel1[:-1]))
                    if len(q1) == 100:
                        qB1 = round(sum(q1) / len(q1) / 2000) * 2000
                        print(f"Using {qB1} as bound for side 1 small primes")
                    if len(q2) == 100:
                        qB2 = round(sum(q2) / len(q2) / 2000) * 2000
                        print(f"Using {qB2} as bound for side 1 small primes")
                    if qB1 is not None:
                        q1co.append(product([_l for _l in rel1 if _l < qB1]))
                    if qB2 is not None:
                        q1q2co.append(product([_l for _l in rel1 if _l < qB2]))
                else:
                    q1co.append(product(rel1[:-1]))
                    if len(p1) == 100:
                        qB1 = round(sum(p1) / len(p1) / 2000) * 2000
                        print(f"Using {qB1} as bound for side 0 small primes")
                    if len(p2) == 100:
                        qB2 = round(sum(p2) / len(p2) / 2000) * 2000
                        print(f"Using {qB2} as bound for side 0 small primes")
                    if qB1 is not None:
                        q1co.append(product([_l for _l in rel0 if _l < qB1]))
                    if qB2 is not None:
                        q1q2co.append(product([_l for _l in rel0 if _l < qB2]))
                b0 = product(rel0).bit_length()
                b1 = product(rel1).bit_length()
                sum0 += b0
                sum1 += b1
                count += 1
    print(count, "relations")
    if side == 0:
        print(f"q+{sum0 / count:.2f} bits for norm0")
        print(f"{sum1 / count:.2f} bits for norm1")
    else:
        print(f"{sum0 / count:.2f} bits for norm0")
        print(f"q+{sum1 / count:.2f} bits for norm1")
    xabs = [abs(x) for x in xs]
    yabs = [abs(y) for y in ys]
    for l in [xabs, yabs, p1, p2, p3, q1, q2, q3, q4, q1co, q1q2co]:
        l.sort()
    q1co.reverse()
    q1q2co.reverse()

    print("SIDE 0 FACTORS")
    print("avg max prime", int(sum(p1) / len(p1)), "range 10/90/95/98/99", deciles(p1))
    print("avg 2nd prime", int(sum(p2) / len(p2)), "range 10/90/95/98/99", deciles(p2))
    print("avg 3rd prime", int(sum(p3) / len(p3)), "range 10/90/95/98/99", deciles(p3))
    print("SIDE 1 FACTORS")
    print("avg max prime", int(sum(q1) / len(q1)), "range 10/90/95/98/99", deciles(q1))
    print("avg 2nd prime", int(sum(q2) / len(q2)), "range 10/90/95/98/99", deciles(q2))
    print("avg 3rd prime", int(sum(q3) / len(q3)), "range 10/90/95/98/99", deciles(q3))
    print("avg 4th prime", int(sum(q4) / len(q4)), "range 10/90/95/98/99", deciles(q4))
    print(f"Average smooth part on side {1 - side}")
    print(
        f"avg smooth(<{qB1})",
        int(sum(q1co) / len(q1co)),
        "range 10/90/95/98/99",
        deciles(q1co),
    )
    print(
        f"avg smooth(<{qB2})",
        int(sum(q1q2co) / len(q1q2co)),
        "range 10/90/95/98/99",
        deciles(q1q2co),
    )

    print("Point locations")
    print("avg X", int(sum(xabs) / len(xabs)), "range 10/90/95/98/99", deciles(xabs))
    print("avg Y", int(sum(yabs) / len(yabs)), "range 10/90/95/98/99", deciles(yabs))
    print("Covariance matrix:")
    xys = numpy.array([xs, ys], dtype=numpy.float64)
    cov = (xys @ xys.T) / len(xs)
    print(cov)
    area = 2 * deciles(xabs)[-2] * deciles(yabs)[-2]
    skew = math.sqrt(cov[0, 0] / cov[1, 1])
    print(f"Effective skew = {skew:.3f}")
    print(f"Sieve area (95%) = {area:.3g}")


def deciles(l):
    assert len(l) > 100
    return (
        l[len(l) // 10],
        l[-len(l) // 10],
        l[-len(l) // 20],
        l[-len(l) // 50],
        l[-len(l) // 100],
    )


if __name__ == "__main__":
    main()
