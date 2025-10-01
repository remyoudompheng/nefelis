import argparse
import logging
import tempfile

from nefelis import deg2
from nefelis import deg3
from nefelis import fp2


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

    sieve_args = argp.add_argument_group("Sieve options")
    sieve_args.add_argument("--ncpu", type=int, help="CPU threads for factoring")
    sieve_args.add_argument(
        "--nogpufactor", action="store_true", help="Don't perform trial division on GPU"
    )

    linalg_args = argp.add_argument_group("Linear algebra options")
    linalg_args.add_argument(
        "--blockw", default=1, type=int, help="Use Block Wiedemann with size m=ARG n=1"
    )

    argp.add_argument("METHOD", choices=("deg2", "deg3", "fp2"))
    argp.add_argument("N", type=int)
    argp.add_argument(
        "ARGS", nargs="*", type=str, help="Arguments for discrete logarithm"
    )
    args = argp.parse_args()

    # Logger names should have length <= 6 (poly, sieve, linalg, dlog)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        force=True,
        style="{",
        format="{relativeCreatedSecs: >9.3f}s {levelname[0]} {name:<6s} {message}",
    )

    def add_relative_seconds(record):
        record.relativeCreatedSecs = record.relativeCreated / 1000.0
        return True

    logging.getLogger().handlers[0].addFilter(add_relative_seconds)

    if args.WORKDIR is None:
        logging.getLogger("main").info("Creating temporary directory for results")
        with tempfile.TemporaryDirectory(prefix="nefelis") as tmpdir:
            args.WORKDIR = tmpdir
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
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
