from __future__ import annotations

import io
import logging

from fem_inhouse.cli import StructuredFormatter


def _render(message: str, extra: dict[str, object] | None) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter("%(levelname)s %(message)s"))
    logger = logging.getLogger(f"test.{message}.{extra}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info(message, extra=extra)
    return stream.getvalue().strip()


def test_structured_fields_reach_the_formatted_line() -> None:
    # Without this, --verbose printed a bare "Newton iteration" and a long
    # campaign could not be followed.
    rendered = _render(
        "Newton iteration",
        {"event": "newton_iteration", "increment": 3, "iteration": 2},
    )

    assert "increment=3" in rendered
    assert "iteration=2" in rendered
    assert rendered.startswith("INFO Newton iteration")


def test_floats_are_rendered_compactly() -> None:
    rendered = _render("Newton iteration", {"relative_residual": 1.2340000001e-05})

    assert "relative_residual=1.234e-05" in rendered


def test_a_record_without_extras_is_left_alone() -> None:
    assert _render("plain message", None) == "INFO plain message"


def test_standard_record_attributes_are_not_echoed() -> None:
    rendered = _render("Newton iteration", {"increment": 1})

    for noise in ("levelno=", "pathname=", "msg=", "args=", "lineno="):
        assert noise not in rendered
