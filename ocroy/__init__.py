from ocroy.parser import parse_args

__version__ = "0.0.1"


def main() -> None:
    args = parse_args()
    args.func(args)
