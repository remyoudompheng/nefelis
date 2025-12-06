import argparse
import logging
import tempfile

from nefelis import deg2
from nefelis import deg3
from nefelis import fp2
from nefelis import factor
import nefelis.logging


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument("--workdir", dest="WORKDIR", help="Path to working directory")

    polyselect_args = argp.add_argument_group("Polynomial selection options")
    polyselect_args.add_argument(
        "--nosm",
        action="store_true",
        help="Choose simple polynomials to avoid Schirokauer maps",
    )

    def gpu_id(s: str) -> list[int]:
        return [int(x) for x in s.split(",")]

    sieve_args = argp.add_argument_group("Sieve options")
    sieve_args.add_argument("--gpu", type=gpu_id, help="List of GPU devices to be used")
    sieve_args.add_argument("--ncpu", type=int, help="CPU threads for factoring")
    sieve_args.add_argument(
        "--lowcpu", action="store_true", help="Choose parameters reducing CPU usage"
    )
    sieve_args.add_argument(
        "--nogpufactor", action="store_true", help="Don't perform trial division on GPU"
    )
    sieve_args.add_argument(
        "--poly", help="Use a custom polynomial (Cado-NFS .poly format)"
    )

    linalg_args = argp.add_argument_group("Linear algebra options")
    linalg_args.add_argument(
        "--blockw", type=int, help="Use Block Wiedemann with size m=ARG n=1"
    )

    factor_args = argp.add_argument_group("Factoring options")
    factor_args.add_argument(
        "--snfs", action="store_true", help="Use Special number field sieve"
    )
    factor_args.add_argument(
        "--parambias", type=int, default=0, help="Use parameters for size N+parambias"
    )

    argp.add_argument("METHOD", choices=("deg2", "deg3", "fp2", "factor"))
    argp.add_argument("N", type=int)
    argp.add_argument(
        "ARGS", nargs="*", type=str, help="Arguments for discrete logarithm"
    )
    args = argp.parse_args()

    # Logger names should have length <= 6 (poly, sieve, linalg, dlog)
    level = logging.DEBUG if args.verbose else logging.INFO
    nefelis.logging.setup(level)

    if args.WORKDIR is None:
        with tempfile.TemporaryDirectory(prefix="nefelis") as tmpdir:
            args.WORKDIR = tmpdir
            logging.getLogger("main").info(
                f"Creating temporary directory {tmpdir} for results"
            )
            main_impl(args)
    else:
        main_impl(args)


def main_impl(args):
    match args.METHOD:
        case "deg2":
            deg2.sieve.main_impl(args)
            deg2.linalg.main_impl(args)
            if args.ARGS:
                deg2.dlog.main_impl(args)
        case "deg3":
            deg3.sieve.main_impl(args)
            deg3.linalg.main_impl(args)
            if args.ARGS:
                deg3.dlog.main_impl(args)
        case "fp2":
            fp2.sieve.main_impl(args)
            fp2.linalg.main_impl(args)
            if args.ARGS:
                fp2.dlog.main_impl(args)
        case "factor":
            factor.sieve.main_impl(args)
            factor.linalg.main_impl(args)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
