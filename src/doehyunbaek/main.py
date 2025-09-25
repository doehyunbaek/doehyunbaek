from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from typing import Sequence

try:
    __version__ = version("doehyunbaek")
except PackageNotFoundError:
    __version__ = "0.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doehyunbaek",
        description="Doehyun’s multipurpose CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # Register subcommands here
    from .anon_to_zenodo import register_subcommand as register_anon_to_zenodo
    from .devc_to_docker import register_subcommand as register_devc_to_docker

    register_anon_to_zenodo(sub)
    register_devc_to_docker(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "_handler", None)
    if callable(handler):
        return handler(args)

    # No subcommand → show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
