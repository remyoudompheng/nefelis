import argparse
import logging
import tempfile

from nefelis import deg2
from nefelis import deg3


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")

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
        "--blockw", type=int, help="Use Block Wiedemann with size m=ARG n=1"
    )

    argp.add_argument("METHOD", choices=("deg2", "deg3"))
    argp.add_argument("N", type=int)
    argp.add_argument("WORKDIR", nargs="?")
    args = argp.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.WORKDIR is None:
        logging.info("Creating temporary directory for results")
        with tempfile.TemporaryDirectory(prefix="nefelis") as tmpdir:
            args.WORKDIR = tmpdir
            main_impl(args)
    else:
        main_impl(args)


def main_impl(args):
    if args.METHOD == "deg2":
        deg2.sieve.main_impl(args)
        deg2.linalg.main_impl(args)
    else:
        deg3.sieve.main_impl(args)
        deg3.linalg.main_impl(args)


if __name__ == "__main__":
    main()
