"""llama-mtmd-eval CLI.

Subcommands:
  llama    run llama-mtmd-cli for selected cases, grade vs stored baselines (the
           regression gate; exits nonzero on any failure). Base env, no torch.
  hf       run the HF reference model(s) to produce/refresh baselines. Needs the
           family's extra synced: `uv sync --extra hf-deepseek|hf-unlimited`.
  compare  run BOTH llama and HF for selected cases, side by side (needs an HF env).
  report   re-render a saved results json.
  score    ad-hoc: score an OCR text file against a ground-truth file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cases as cases_mod
from . import models as models_mod
from . import report as report_mod
from .config import load_config


def _select(items, keys, attr):
    if not keys or keys == "all":
        return items
    wanted = {k.strip() for k in keys.split(",")}
    return [it for it in items if getattr(it, attr) in wanted]


def _filter_cases(all_cases, model_sel, case_sel):
    picked = _select(all_cases, model_sel, "model")
    if case_sel and case_sel != "all":
        needles = [n.strip() for n in case_sel.split(",")]
        picked = [c for c in picked if any(n in c.label or n in c.image for n in needles)]
    return picked


def _load(args, *, need_hf: bool = False):
    overrides = {
        "llama_bin": getattr(args, "llama_bin", None),
        "gguf_dir": getattr(args, "gguf_dir", None),
        "hf_dir": getattr(args, "hf_dir", None),
        "device": getattr(args, "device", None),
    }
    cfg = load_config(Path(args.config) if args.config else None, overrides, need_hf=need_hf)
    models = models_mod.load_models()
    all_cases = cases_mod.load_cases(known_models=set(models))
    picked = _filter_cases(all_cases, args.model, args.case)
    if not picked:
        raise SystemExit(f"no cases match --model {args.model!r} --case {args.case!r}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return cfg, models, picked


def _emit(results, fmt, out):
    print(report_mod.render(results, fmt))
    if out:
        Path(out).write_text(report_mod.to_json(results), encoding="utf-8")


def _preflight(spec, case, cfg, source) -> str:
    """Missing-file report for a case, before spending minutes running it."""
    paths = [("image", cfg.image(case.image)),
             ("ground-truth", cfg.ground_truth(case.ground_truth))]
    if source == "llama":
        paths += [("llama-bin", cfg.llama_bin),
                  ("model", cfg.gguf(spec.llama.gguf)),
                  ("mmproj", cfg.gguf(spec.llama.mmproj))]
    else:
        paths += [("hf-model", cfg.hf_model(spec.hf.dir))]
    missing = [f"{label} not found: {p}" for label, p in paths if not p.exists()]
    return "; ".join(missing)


def _run_case(spec, case, cfg, source, runner) -> "Result":
    """Run one case; a failure becomes an ERROR row instead of killing the run."""
    from .evaluate import error_result
    problem = _preflight(spec, case, cfg, source)
    if not problem:
        try:
            return runner()
        except (RuntimeError, OSError) as e:
            problem = str(e)
    print(f"# ERROR {source} {case.model} -- {case.label}: {problem.splitlines()[0]}",
          file=sys.stderr)
    return error_result(spec, case, source, problem)


def _split_by_hf_env(models, picked):
    """Cases whose HF env matches the synced venv vs the rest (skipped, not fatal)."""
    from .hf_runner import active_env
    active = active_env()
    runnable = [c for c in picked if models[c.model].hf.env == active]
    for c in picked:
        env = models[c.model].hf.env
        if env != active:
            print(f"# skip hf {c.model} -- {c.label}: needs `uv sync --extra {env}` "
                  f"(active: {active or 'none'})", file=sys.stderr)
    return runnable, active


def cmd_llama(args) -> int:
    from .evaluate import evaluate_llama
    cfg, models, picked = _load(args)
    results = []
    for c in picked:
        print(f"# llama {c.model} -- {c.label} ({c.image})", file=sys.stderr)
        spec = models[c.model]
        results.append(_run_case(spec, c, cfg, "llama",
                                 lambda: evaluate_llama(spec, c, cfg)))
    _emit(results, args.format, args.out)
    return 0 if report_mod.gate(results) else 1


def cmd_hf(args) -> int:
    from .evaluate import evaluate_hf
    cfg, models, picked = _load(args, need_hf=True)
    device = args.device or cfg.device
    runnable, active = _split_by_hf_env(models, picked)
    if not runnable:
        envs = sorted({models[c.model].hf.env for c in picked})
        raise SystemExit(f"no selected model runs in the active env "
                         f"({active or 'none'}). Sync one of: "
                         + " | ".join(f"`uv sync --extra {e}`" for e in envs))
    results = []
    for c in runnable:
        print(f"# hf {c.model} -- {c.label} ({c.image})", file=sys.stderr)
        spec = models[c.model]
        results.append(_run_case(spec, c, cfg, "hf",
                                 lambda: evaluate_hf(spec, c, cfg, device=device)))
    _emit(results, args.format, args.out)
    return 0


def cmd_compare(args) -> int:
    from .evaluate import evaluate_hf, evaluate_llama
    cfg, models, picked = _load(args, need_hf=True)
    device = args.device or cfg.device
    hf_runnable, _ = _split_by_hf_env(models, picked)
    results = []
    for c in picked:
        print(f"# compare {c.model} -- {c.label} ({c.image})", file=sys.stderr)
        spec = models[c.model]
        results.append(_run_case(spec, c, cfg, "llama",
                                 lambda: evaluate_llama(spec, c, cfg)))
        if c in hf_runnable:
            results.append(_run_case(spec, c, cfg, "hf",
                                     lambda: evaluate_hf(spec, c, cfg, device=device)))
    _emit(results, args.format, args.out)
    return 0 if report_mod.gate(results) else 1


def cmd_report(args) -> int:
    from .evaluate import Result
    data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    results = [Result(**d) for d in data]
    print(report_mod.render(results, args.format))
    return 0


def cmd_score(args) -> int:
    from .scoring import score
    s = score(Path(args.ocr).read_text(encoding="utf-8"),
              Path(args.ground_truth).read_text(encoding="utf-8"),
              strip_grounding_markup=args.strip_grounding)
    print(f"CER={s.cer:.4f}  chrF={s.chrf:.2f}")
    return 0


def _add_common(p, *, device=False):
    p.add_argument("--model", default="all", help="v1,v2,unlimited or all")
    p.add_argument("--case", default="all", help="filter by label/image substring, or all")
    p.add_argument("--config", default=None, help="path to config.toml")
    p.add_argument("--llama-bin", default=None)
    p.add_argument("--gguf-dir", default=None)
    p.add_argument("--hf-dir", default=None)
    p.add_argument("--format", default="console", choices=["console", "md", "json"])
    p.add_argument("--out", default=None, help="also write results json to this path")
    if device:
        p.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mtmd-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("llama", help="run llama.cpp, grade vs baselines (regression gate)")
    _add_common(p)
    p.set_defaults(func=cmd_llama, device=None)

    p = sub.add_parser("hf", help="run HF reference model(s)")
    _add_common(p, device=True)
    p.set_defaults(func=cmd_hf)

    p = sub.add_parser("compare", help="run llama + HF side by side")
    _add_common(p, device=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report", help="re-render a saved results json")
    p.add_argument("from_json", metavar="results.json")
    p.add_argument("--format", default="console", choices=["console", "md", "json"])
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("score", help="score an OCR text file vs a ground-truth file")
    p.add_argument("ocr")
    p.add_argument("ground_truth")
    p.add_argument("--strip-grounding", action="store_true")
    p.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
