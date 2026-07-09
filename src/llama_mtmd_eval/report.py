"""Render evaluation Results as a console table, markdown, or json."""
from __future__ import annotations

import json

from .evaluate import Result

_COLS = ("model", "label", "source", "CER", "chrF", "baseline", "band", "verdict")


def gate(results: list[Result]) -> bool:
    """The regression verdict: llama rows must all pass (HF rows are references).
    Used for both the rendered Overall line and the process exit code."""
    gated = [r for r in results if r.source == "llama"]
    return all(r.passed for r in gated)


def _overall(results: list[Result]) -> str:
    llama_rows = [r for r in results if r.source == "llama"]
    if llama_rows:
        passed = sum(r.passed for r in llama_rows)
        line = f"Overall: {passed}/{len(llama_rows)} llama cases passed"
        if len(llama_rows) != len(results):
            line += " (HF rows are references, not gated)"
        return line + ("" if gate(results) else "  -> FAIL")
    passed = sum(r.passed for r in results)
    return f"Overall: {passed}/{len(results)} within stored baselines (reference run, not gated)"


def _rows(results: list[Result]) -> list[tuple[str, ...]]:
    rows = []
    for r in results:
        if r.error:
            verdict = "ERROR"
            cer = chrf = "-"
        else:
            verdict = "PASS" if r.passed else "FAIL"
            cer, chrf = f"{r.cer:.4f}", f"{r.chrf:.2f}"
        if r.provisional:
            verdict += " *prov"
        if r.approximate:
            verdict += " ~approx"
        rows.append((
            r.model, r.label, r.source, cer, chrf,
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
    errors = [r for r in results if r.error]
    lines.append("")
    for r in errors:
        lines.append(f"ERROR {r.model} -- {r.label}: {r.error.splitlines()[0]}")
    lines.append(_overall(results))
    return "\n".join(lines)


def markdown(results: list[Result]) -> str:
    header = "| " + " | ".join(_COLS) + " |"
    sep = "|" + "|".join(["---"] * len(_COLS)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in _rows(results)]
    return "\n".join([header, sep, *body, "", f"**{_overall(results)}**"])


def to_json(results: list[Result]) -> str:
    return json.dumps([r.as_dict() for r in results], indent=2)


def render(results: list[Result], fmt: str = "console") -> str:
    return {"console": console, "md": markdown, "json": to_json}[fmt](results)
