import sys
import argparse


from numpy import linalg as la
import numpy as np


def parse_input_args(program, desc):
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8 or later to run this script\n")
        sys.exit(1)

    p = argparse.ArgumentParser(
        prog=program,
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("required_positional_arg", help="desc")
    p.add_argument("required_int", type=int, help="req number")
    p.add_argument("--on", action="store_true", help="include to enable")
    p.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="increase output verbosity (default: %(default)s)",
    )

    group1 = p.add_mutually_exclusive_group(required=True)
    group1.add_argument("--enable", action="store_true")
    group1.add_argument("--disable", action="store_false")

    return p.parse_args()


parse_input_args(1, 1)
