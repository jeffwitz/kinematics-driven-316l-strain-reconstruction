"""Allow ``python -m fem_inhouse`` to invoke the project CLI."""

from fem_inhouse.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
