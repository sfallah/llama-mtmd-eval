"""Run an HF reference OCR model and return its raw output.

All three families are DeepSeek-OCR-style custom-code models loaded via
`AutoModel(..., trust_remote_code=True)` and driven with `model.infer(tokenizer,
prompt, image_file, base_size, image_size, crop_mode, eval_mode, **kwargs)`.

torch/transformers are imported lazily so the base (llama/compare) env never needs
them. Because the two families pin incompatible transformers versions, this module
guards at runtime: it derives the active env from `transformers.__version__` and
refuses a model whose `hf.env` doesn't match, with an actionable `uv sync` message.

The reference regime is **fp32 weights + the model's internal autocast-bf16** (how the
canonical scores were measured): loading bf16 *weights* is lossier and derails greedy
decode on fragile images. So we load fp32 weights and let `model.infer()`'s hardcoded
`autocast("cuda", bf16)` do the compute. On a non-CUDA box a device shim routes
`.cuda()`/`autocast("cuda")` -> CPU (an APPROXIMATE score, not an authoritative baseline).

PYTORCH_JIT is disabled so the custom code's `@torch.jit.script` (quick_gelu) runs eager
instead of nvrtc-JIT-compiling a kernel — nvrtc rejects newer GPU archs (e.g. GB10 /
Blackwell: "invalid value for --gpu-architecture"). Must be set before torch imports.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_JIT", "0")

from .models import ModelSpec

# transformers version -> which conflict-extra env is active.
_ENV_BY_VERSION_PREFIX = {"4.46": "hf-deepseek", "4.57": "hf-unlimited"}


@dataclass
class HFResult:
    text: str
    device: str
    approximate: bool   # True when not run on CUDA (score is indicative, not a baseline)


def _active_env() -> str | None:
    import transformers
    for prefix, env in _ENV_BY_VERSION_PREFIX.items():
        if transformers.__version__.startswith(prefix):
            return env
    return None


def _resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cpu"   # mps/cpu -> CPU shim path (vanilla custom code is unreliable on MPS)


def _install_cpu_shim() -> None:
    """Route the model's hardcoded CUDA calls to CPU so the vanilla code runs off-GPU."""
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
    torch.cuda.is_available = lambda: False          # type: ignore[assignment]
    _orig_autocast = torch.autocast

    def _cpu_autocast(device_type="cpu", **k):
        return _orig_autocast("cpu", **k)

    torch.autocast = _cpu_autocast                   # type: ignore[assignment]


def run_hf(spec: ModelSpec, model_dir: Path, image: Path, device: str = "auto") -> HFResult:
    active = _active_env()
    if spec.hf.env != active:
        raise SystemExit(
            f"model '{spec.key}' needs env '{spec.hf.env}' but the active transformers "
            f"env is '{active or 'unknown'}'. Re-sync: `uv sync --extra {spec.hf.env}`"
        )

    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _resolve_device(device)
    if dev != "cuda":
        _install_cpu_shim()

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tok.pad_token = tok.eos_token
    # fp32 WEIGHTS on both paths (the reference regime); the model's internal
    # autocast does bf16 compute. bf16 weights derail greedy decode on fragile images.
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, use_safetensors=True,
        attn_implementation="eager", torch_dtype=torch.float32,
    ).eval()
    model = model.cuda() if dev == "cuda" else model.to("cpu")

    hf = spec.hf
    with tempfile.TemporaryDirectory() as out:  # infer() always mkdirs output_path
        text = model.infer(
            tok, prompt=hf.prompt, image_file=str(image), output_path=out,
            base_size=hf.base_size, image_size=hf.image_size, crop_mode=hf.crop_mode,
            eval_mode=hf.eval_mode, **hf.infer_kwargs,
        )
    return HFResult(text=text.strip(), device=dev, approximate=(dev != "cuda"))
