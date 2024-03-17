from ocroy.parser import parse_args

__version__ = "0.1.0"


def main() -> None:
    args = parse_args()
    args.func(args)
