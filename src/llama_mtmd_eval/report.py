"""Render evaluation Results as a console table, markdown, or json."""
from __future__ import annotations

import json

from .evaluate import Result

_COLS = ("model", "label", "source", "CER", "chrF", "baseline", "band", "verdict")


def _rows(results: list[Result]) -> list[tuple[str, ...]]:
    rows = []
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        if r.provisional:
            verdict += " *prov"
        if r.approximate:
            verdict += " ~approx"
        rows.append((
            r.model, r.label, r.source,
            f"{r.cer:.4f}", f"{r.chrf:.2f}",
            f"{r.baseline_cer:.4f}/{r.baseline_chrf:.2f}",
            f"<={r.cer_max:.4f} / >={r.chrf_min:.2f}",
            verdict,
        ))
    return rows


def console(results: list[Result]) -> str:
    rows = [_COLS] + _rows(results)
    widths = [max(len(row[i]) for row in rows) for i in range(len(_COLS))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(_COLS))))
    passed = sum(r.passed for r in results)
    lines.append("")
    lines.append(f"Overall: {passed}/{len(results)} passed"
                 + ("" if passed == len(results) else "  -> FAIL"))
    return "\n".join(lines)


def markdown(results: list[Result]) -> str:
    header = "| " + " | ".join(_COLS) + " |"
    sep = "|" + "|".join(["---"] * len(_COLS)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in _rows(results)]
    passed = sum(r.passed for r in results)
    return "\n".join([header, sep, *body, "", f"**Overall: {passed}/{len(results)} passed**"])


def to_json(results: list[Result]) -> str:
    return json.dumps([r.as_dict() for r in results], indent=2)


def render(results: list[Result], fmt: str = "console") -> str:
    return {"console": console, "md": markdown, "json": to_json}[fmt](results)
