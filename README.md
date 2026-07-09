# llama-mtmd-eval

Regression + evaluation harness for llama.cpp `mtmd` OCR models, checked against HF
reference baselines. Starts with three DeepSeek-OCR-family models and is built to grow:

| key | model | HF env |
|---|---|---|
| `v1` | DeepSeek-OCR | `hf-deepseek` |
| `v2` | DeepSeek-OCR-2 | `hf-deepseek` |
| `unlimited` | Unlimited-OCR (Baidu) | `hf-unlimited` |

It runs `llama-mtmd-cli` per case, scores the OCR against a ground-truth transcript
(NFC-normalize → fuzzy local-align → CER + chrF), and grades it against per-case
baselines within tolerances. The same scoring runs the HF reference models to
*produce* those baselines (on CUDA — the Spark box — for authoritative numbers).

## The two-environment split

The HF reference models pin incompatible `transformers` versions, so they can't share
one env:

- **v1 + v2** (custom-code) → `transformers==4.46.3` → extra **`hf-deepseek`**
- **Unlimited-OCR** → `transformers==4.57.1` → extra **`hf-unlimited`**

They're declared conflicting in `pyproject.toml` (`[tool.uv] conflicts`), so one
lockfile holds both and you `uv sync --extra <one>` to switch. The **base env is
torch-free** — the regression gate (`llama` / `report` / `score`) needs no torch at
all and runs anywhere.

```
uv sync                          # base: llama gate + scoring, no torch
uv sync --extra hf-deepseek      # + torch + transformers 4.46.3  (v1, v2 HF)
uv sync --extra hf-unlimited     # + torch + transformers 4.57.1  (Unlimited HF)
```

On a Linux CUDA box (the DGX Spark, GB10) torch resolves from the cu128 index; on
macOS it falls back to PyPI (MPS/CPU, HF scores flagged *approximate*).

## Setup

```
cp config.example.toml config.toml     # edit: llama_bin, gguf_dir, hf_dir, device
```

Any value can also come from an env var (`MTMD_LLAMA_BIN`, `MTMD_GGUF_DIR`,
`MTMD_HF_DIR`, `MTMD_DEVICE`) or a CLI flag (`--llama-bin`, …). Precedence: CLI > env >
`config.toml` > default. GGUF and HF models live outside this repo; the test images +
ground truths are bundled under `data/`.

## Usage

```
# regression gate — run llama.cpp, grade vs baselines (exit nonzero on any fail)
uv run mtmd-eval llama --model all
uv run mtmd-eval llama --model v1 --case webpage --format md

# HF references — produce/refresh baselines (needs the family's extra synced)
uv sync --extra hf-deepseek
uv run mtmd-eval hf --model v1,v2 --device cuda --out hf-deepseek.json
uv sync --extra hf-unlimited
uv run mtmd-eval hf --model unlimited --device cuda --out hf-unlimited.json

# llama vs HF side by side
uv run mtmd-eval compare --model v1 --device cuda --format md

# ad-hoc: score any OCR text file against a ground truth
uv run mtmd-eval score out.txt data/ground_truth/test-1-ground-truth.txt
```

`--model` takes `all` or a comma list of keys; `--case` filters by label/image
substring. The `hf`/`compare` commands skip models whose env isn't synced (printing
the exact `uv sync --extra …` to run) and error only if nothing is runnable. A case
that fails to run becomes an ERROR row instead of aborting the remaining cases.

## Run on the Spark box (GB10, CUDA)

```
git clone https://github.com/sfallah/llama-mtmd-eval && cd llama-mtmd-eval
cp config.example.toml config.toml           # point at the Spark llama build + models
uv run mtmd-eval llama --model all            # regression gate (llama built with CUDA)
uv sync --extra hf-deepseek  && uv run mtmd-eval hf --model v1,v2 --device cuda --out hf-deepseek.json
uv sync --extra hf-unlimited && uv run mtmd-eval hf --model unlimited --device cuda --out hf-unlimited.json
```

The HF numbers from CUDA are the authoritative baselines to paste into
`cases/cases.toml`. All bundled baselines (including `webpage (tall)`) are already
HF CUDA references; the webpage case guards the llama.cpp deepseek-ocr fuse_row
tile-drop fix and fails if right-column tiles get dropped again.

## Adding models / cases

Everything data-driven lives in TOML:

- `cases/models.toml` — one entry per family: the llama side (`gguf`, `mmproj`,
  `prompt`, `n_predict`, `n_ctx`, `dry`, `strip_grounding`) and the HF side (`dir`,
  `env`, `prompt`, `image_size`, `crop_mode`, `strip_grounding`, `infer_kwargs`, …).
- `cases/cases.toml` — a case is a `model` key + `image` + `ground_truth` + `baseline`
  {cer,chrf} + `tol` {cer,chrf}. Mark `provisional = true` if the baseline isn't yet an
  HF reference.

Drop new images/ground-truths under `data/` and add a `[[cases]]` block.

## Tests

```
uv run --group dev pytest        # torch-free unit tests for the scoring pipeline
```
