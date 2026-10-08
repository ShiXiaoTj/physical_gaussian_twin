"""Top-level `pgtwin` CLI dispatcher."""

from __future__ import annotations

import sys

from pgtwin import __version__

_COMMANDS = {
    "calibrate": "pgtwin.cli.calibrate",
    "evaluate": "pgtwin.cli.evaluate",
    "reconstruct": "pgtwin.cli.reconstruct",
}

_USAGE = f"""pgtwin {__version__} — Physical Gaussian Digital Twin

usage: pgtwin <command> [args...]

commands:
  calibrate    calibrate physics parameters from a real episode
  evaluate     re-evaluate a saved experiment directory
  reconstruct  import/inspect a Gaussian scene

run `pgtwin <command> --help` for command options.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    command = argv[0]
    if command not in _COMMANDS:
        print(f"unknown command '{command}'\n\n{_USAGE}", file=sys.stderr)
        return 2
    import importlib

    module = importlib.import_module(_COMMANDS[command])
    return int(module.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
