# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

*Train agent skills like you train neural networks — with epochs, (mini-)batchsize, learning rates, and validation gates — but without touching model weights.*

[![Project Page](https://img.shields.io/badge/Project%20Page-SkillOpt-8dbb3c)](https://microsoft.github.io/SkillOpt/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.23904) [![Project Video](https://img.shields.io/badge/Project%20Video-Watch%20Demo-ff0000)](https://youtu.be/JUBMDTCiM0M) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Modern agent skills are usually hand-crafted, generated one-shot by a strong
LLM, or evolved through loosely controlled self-revision — none of which
behaves like a deep-learning optimizer for the skill itself, and none of
which reliably improves over its starting point under feedback.

**SkillOpt treats the skill document as the trainable state of a frozen
agent**, and trains it with the discipline that makes weight-space
optimization reproducible. A separate optimizer model turns scored rollouts
into bounded add / delete / replace edits on a single skill document; a
candidate edit is accepted only when it strictly improves a held-out
validation score. A textual learning-rate budget, a rejected-edit buffer,
and an epoch-wise slow / meta update make skill training stable while
adding **zero inference-time model calls** at deployment.

The deployed artifact is a compact `best_skill.md` (typically 300–2,000
tokens) that runs against the unchanged target model. Across **six
benchmarks, seven target models, and three execution harnesses** (direct
chat, Codex CLI, Claude Code CLI), SkillOpt is best or tied-best on **all
52 evaluated (model, benchmark, harness) cells** and on GPT-5.5 lifts the
average no-skill accuracy by **+23.5 points in direct chat, +24.8 inside
the Codex agentic loop, and +19.1 inside Claude Code**. Optimized skill
artifacts transfer across model scales, between Codex and Claude Code
harnesses, and to nearby benchmarks without further optimization.

For the full method, ablations, and per-cell results see the [paper](https://arxiv.org/abs/2605.23904); for a visual walkthrough of the loop see the [project page](https://microsoft.github.io/SkillOpt/); for deeper API / backend / benchmark docs see [`docs/`](docs/).

## 🎬 Demo Video

https://github.com/user-attachments/assets/eb12d3bc-371c-467f-904d-91b61f339ed7

<p align="center">
  <a href="https://youtu.be/JUBMDTCiM0M"><b>▶ Watch the full demo on YouTube</b></a>
</p>

---

## Install

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# For the ALFWorld benchmark (optional):
uv pip install -e ".[alfworld]"
alfworld-download
```

### Configure API Credentials

`.env.example` ships preconfigured for **Claude via the Claude CLI**
(`OPTIMIZER_BACKEND` / `TARGET_BACKEND` = `claude_chat`, model
`claude-sonnet-4-6`). For that path you don't need to edit anything — just copy
and source it, then log in to the CLI (see below):

```bash
cp .env.example .env
source .env
```

To use a different backend (Azure OpenAI, Qwen, MiniMax), edit `.env`: set
`OPTIMIZER_BACKEND` / `TARGET_BACKEND` and that backend's credentials — see the
sections below. You can also override the backend per run with `--backend` on
any script.

#### Anthropic Claude *(via Claude CLI — no API key needed)*

The `claude_chat` backend doesn't call the Anthropic API directly — it shells
out to the [`claude` CLI](https://docs.claude.com/claude-code) (`claude -p`).
Authentication therefore piggybacks on the CLI's own login, so there is **no
API key to manage** and nothing secret in `.env`. The backend itself is already
selected by `.env.example` (see [Configure API Credentials](#configure-api-credentials)
above) — all that's left is the CLI login.

**Step 1 — Install the CLI and log in once** (skip if you already have it):

```bash
claude auth login          # or run `claude` and use /login
claude --version           # sanity check: should print a version
```

**Step 2 — Verify the wiring** with a one-line smoke test:

```bash
OPTIMIZER_BACKEND=claude_chat TARGET_BACKEND=claude_chat python -c "
from skillopt.model import set_target_backend, chat_target
set_target_backend('claude_chat')
text, usage = chat_target('You are a calculator. Reply with only the number.', 'What is 21 + 21?')
print('reply =', text.strip(), '| usage =', usage)
"
# Expected: reply = 42 | usage = {...}
```

If you see `reply = 42`, the CLI, login, and backend are all wired correctly.

> **Note:** Do **not** set `ANTHROPIC_API_KEY` when relying on the CLI
> subscription login — if it is present in the environment, the CLI bills
> against the API key instead of your login. Set it only if you specifically
> want API-key billing.

#### Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
# Option 1: API key auth
export AZURE_OPENAI_API_KEY="your-key"
# Option 2: Azure CLI auth (no API key needed)
export AZURE_OPENAI_AUTH_MODE="azure_cli"
```

> **Note:** `AZURE_OPENAI_ENDPOINT` is required for all three modes (`api_key`, `azure_cli`, `openai_compatible`). Without it, all LLM calls will fail.

#### OpenAI-compatible endpoints

```bash
export AZURE_OPENAI_ENDPOINT="https://api.openai.com/v1"
export AZURE_OPENAI_API_KEY="sk-..."
export AZURE_OPENAI_AUTH_MODE="openai_compatible"
```

This routes all calls through the plain OpenAI Python client (no Azure auth, no `api-version` header).

> **Note:** SkillOpt reuses the `AZURE_OPENAI_*` env var names even in this mode — there is no separate `OPENAI_API_KEY` knob.

#### Qwen *(local vLLM)*

```bash
export QWEN_CHAT_BASE_URL="http://localhost:8000/v1"
export QWEN_CHAT_MODEL="Qwen/Qwen3.5-4B"
```

`qwen_chat` can also be used as the optimizer backend. When optimizer and
target should point to different local vLLM services, use the role-specific
settings:

```bash
python scripts/train.py \
    --config configs/searchqa/default.yaml \
    --optimizer_backend qwen_chat \
    --target_backend qwen_chat \
    --optimizer_model Qwen/Qwen3.5-4B \
    --target_model Qwen/Qwen3.5-4B \
    --optimizer_qwen_chat_base_url http://localhost:8001/v1 \
    --target_qwen_chat_base_url http://localhost:8000/v1
```

#### MiniMax

```bash
export MINIMAX_BASE_URL="https://api.minimax.io/v1"
export MINIMAX_API_KEY="..."
export MINIMAX_MODEL="MiniMax-M2.7"
```

---

## Quick Start

### Training

> **What is SearchQA?** A context-grounded question-answering benchmark used as
> the running example below. The agent receives the skill document, a question,
> and context passages, then must return its answer inside `<answer>...</answer>`
> tags (single turn). SkillOpt doesn't fine-tune the model — it iteratively
> rewrites the *skill* so the agent answers more accurately. See
> [Supported Benchmarks](#supported-benchmarks) for the full list.

> ⚠️ **Materialize the data first.** The repo ships only *split manifests*
> (`data/*_id_split/` — lists of example **IDs**, no questions/contexts/answers).
> Pointing `--split_dir` at a `*_id_split/` manifest will run but score 0.0
> everywhere, since the examples have no content. Materialize the runnable split
> into the `split_dir` each config expects first:
>
> ```bash
> uv pip install -e ".[data]"                 # one-time: pulls in datasets + huggingface_hub
> python scripts/materialize_searchqa.py      # → data/searchqa_split/
> python scripts/materialize_livemath.py      # → data/livemathematicianbench_split/
> ```

| Benchmark | Materialize with | Notes |
|---|---|---|
| SearchQA | `python scripts/materialize_searchqa.py` | text join on `lucadiliello/searchqa` |
| LiveMathematicianBench | `python scripts/materialize_livemath.py` | downloads monthly QA files, joins by `<month>:<no>` |
| DocVQA | manual — see [`data/README.md`](data/README.md) | needs page **images** from `lmms-lab/DocVQA` |
| OfficeQA | manual — see [`data/README.md`](data/README.md) | needs supporting **documents** |
| SpreadsheetBench | manual — see [`data/README.md`](data/README.md) | needs **spreadsheet** files |
| ALFWorld | `alfworld-download` (sets `$ALFWORLD_DATA`) | manifest is usable directly as `--split_dir` |

**Claude CLI way** — no endpoint, no API key, no `--*_model` flags (defaults to
`claude-sonnet-4-6`). Just add `--backend claude`:

```bash
# Train on SearchQA with Claude (after materializing data/searchqa_split):
python scripts/train.py \
    --config configs/searchqa/default.yaml \
    --split_dir data/searchqa_split \
    --backend claude

# Other benchmarks (Claude) — each split_dir is also the config default, so it's optional:
python scripts/train.py --config configs/livemathematicianbench/default.yaml --split_dir data/livemathematicianbench_split --backend claude
python scripts/train.py --config configs/alfworld/default.yaml --split_dir data/alfworld_path_split --backend claude
```

Each `--split_dir` above is already the config default, so you can omit it. The
same pattern works for any benchmark — swap the `--config`/`--split_dir` (after
materializing that benchmark's data per the table above). Using Azure OpenAI
instead, drop `--backend claude` and pass the endpoint + models:

```bash
# Train on SearchQA with Azure OpenAI:
python scripts/train.py \
    --config configs/searchqa/default.yaml \
    --split_dir data/searchqa_split \
    --azure_openai_endpoint https://your-resource.openai.azure.com/ \
    --optimizer_model gpt-5.5 \
    --target_model gpt-5.5

# Other benchmarks (Azure):
python scripts/train.py --config configs/livemathematicianbench/default.yaml --split_dir data/livemathematicianbench_split --azure_openai_endpoint https://your-resource.openai.azure.com/ --optimizer_model gpt-5.5 --target_model gpt-5.5
python scripts/train.py --config configs/alfworld/default.yaml --split_dir data/alfworld_path_split --azure_openai_endpoint https://your-resource.openai.azure.com/ --optimizer_model gpt-5.5 --target_model gpt-5.5
```

Key CLI arguments:

| Argument | Description | Example |
|---|---|---|
| `--config` | Benchmark config YAML | `configs/searchqa/default.yaml` |
| `--split_dir` | Path to data split directory | `/path/to/split` |
| `--backend` | Backend shorthand (sets optimizer + target) | `claude` |
| `--azure_openai_endpoint` | Azure OpenAI endpoint URL *(Azure backend only)* | `https://your-resource.openai.azure.com/` |
| `--optimizer_model` | Optimizer model *(optional for Claude; defaults to `claude-sonnet-4-6`)* | `gpt-5.5` |
| `--target_model` | Target model *(optional for Claude; defaults to `claude-sonnet-4-6`)* | `gpt-5.5` |
| `--num_epochs` | Number of training epochs | `4` |
| `--batch_size` | Batch size per step | `40` |
| `--workers` | Parallel rollout workers | `8` |
| `--out_root` | Output directory | `outputs/my_run` |

### Eval Only

Evaluate a trained skill on specific data splits without training.

**Claude CLI way** — just add `--backend claude` (no endpoint, no models):

```bash
# Evaluate the packaged SearchQA skill on one split with Claude:
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill ckpt/searchqa/gpt5.5_skill.md \
  --split valid_unseen \
  --split_dir data/searchqa_split \
  --backend claude

# Evaluate on all splits (train + val + test):
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill ckpt/searchqa/gpt5.5_skill.md \
  --split all \
  --split_dir data/searchqa_split \
  --backend claude
```

Using Azure OpenAI instead, drop `--backend claude` and pass
`--azure_openai_endpoint https://your-resource.openai.azure.com/`.

To evaluate a skill produced by your own training run, replace `--skill` with that run's best-skill path, for example `outputs/my_run/best_skill.md`.

| Split | Description |
|---|---|
| `valid_unseen` | Test set |
| `valid_seen` | Validation set |
| `train` | Training set |
| `all` | All splits combined (default) |

### Output Structure

Each training run writes to a structured output directory:

```
outputs/<run_name>/
├── config.json              # Flattened runtime config
├── history.json             # Per-step training history
├── runtime_state.json       # Resume checkpoint
├── best_skill.md            # Best validated skill document
├── skills/skill_vXXXX.md   # Skill snapshot per step
├── steps/step_XXXX/        # Per-step artifacts (patches, evals)
├── slow_update/epoch_XX/   # Slow update logs
└── meta_skill/epoch_XX/    # Meta skill logs
```

Re-running the same command auto-resumes from the last completed step.

### Pretrained Skill Artifacts

We provide a subset of the paper's main Table 1 GPT-5.5 optimized skills in
[`ckpt/`](ckpt/) as reference artifacts. Use them with `scripts/eval_only.py`
to evaluate the provided skills on a matching data split without re-running
training. See [`ckpt/README.md`](ckpt/README.md) for the full per-benchmark
command. This is the first artifact batch; we plan to continue uploading
the remaining optimized skills and benchmark split manifests as they are
cleaned and verified.

---

## Data Preparation

### Directory layout

SkillOpt expects data in a **split directory** with `train/`, `val/`, `test/` subdirectories, each containing a JSON file (e.g., `items.json`):

```
data/my_split/
├── train/items.json
├── val/items.json
└── test/items.json
```

Each JSON file is an array of task items. The required fields depend on the benchmark. For example, SearchQA items look like:

```json
[
  {
    "id": "unique_item_id",
    "question": "Who wrote the novel ...",
    "context": "[DOC] relevant passage text ...",
    "answers": ["expected answer"]
  }
]
```

See `skillopt/envs/<benchmark>/dataloader.py` for the exact format each benchmark expects.

> **Note:** Most benchmark datasets are not included in this repository — the `data/*_id_split/` directories are **ID manifests only** (stable example IDs per split, no payload). To get a runnable split you materialize the full examples from the upstream source listed in [`data/README.md`](data/README.md) into the `split_dir` path each config expects (e.g. `data/searchqa_split/`). For SearchQA this is automated: `python scripts/materialize_searchqa.py` joins [`data/searchqa_id_split/`](data/searchqa_id_split) (400 / 200 / 1400) against [`lucadiliello/searchqa`](https://huggingface.co/datasets/lucadiliello/searchqa) and writes `data/searchqa_split/`. For other benchmarks, follow the lookup keys in [`data/README.md`](data/README.md).

### Supported Benchmarks

| Benchmark | Type | Config |
|---|---|---|
| SearchQA | QA | `configs/searchqa/default.yaml` |
| ALFWorld | Embodied agent | `configs/alfworld/default.yaml` |
| DocVQA | Document QA | `configs/docvqa/default.yaml` |
| LiveMathematicianBench | Math | `configs/livemathematicianbench/default.yaml` |
| SpreadsheetBench | Code generation | `configs/spreadsheetbench/default.yaml` |
| OfficeQA | Tool-augmented QA | `configs/officeqa/default.yaml` |

---

## Configuration

### Default settings and paper-reproduction knobs

`configs/_base_/default.yaml` is the single source of truth for SkillOpt's
runtime knobs. Out of the box, every included benchmark config inherits
from it and keeps the paper protocol visible: 4 epochs, rollout batch 40,
reflection minibatch 8, textual learning rate 4 with cosine decay, strict
hard validation gating, and slow-update + meta-skill enabled. One detail to
watch is slow-update acceptance: the current `main` default is the newer
post-submission force-accept mode, while the paper protocol and the
paper-aligned skills under `ckpt/` use the gated semantics described in
paper Section 3.6.

### Slow-update acceptance mode

The epoch-boundary slow / meta update can be applied two ways, controlled
by `optimizer.slow_update_gate_with_selection`:

```yaml
optimizer:
  slow_update_gate_with_selection: false   # current main default
```

- **`false`** *(current `main` default)*: force-accept. The
  slow-update guidance is injected into both `current_skill` and
  `best_skill` unconditionally at the epoch boundary. This is the newer
  post-submission behavior on `main`.
- **`true`** *(paper / ckpt-skill reproduction)*: gated, matching paper
  Section 3.6 verbatim. The slow-update candidate is evaluated on the
  selection split and accepted only if it passes the same validation gate
  as a step-level edit. Use this setting when re-running optimization to
  match the paper protocol and the provenance of the provided `ckpt/` skills.

The trainer prints which mode is active at startup
(`[slow update] acceptance=...`). See issue #22 for the discussion that
led to the flag.

### Gate metric (`hard` / `soft` / `mixed`)

The validation gate compares candidate vs. current skills on the selection
split using `gate_metric`:

- **`hard`** *(default, paper)*: exact-match accuracy, strictly greater
  than the current score is required.
- **`soft`**: per-item soft / partial-credit score. Useful when the
  selection split is small (e.g. ≤10 items) and the reward is continuous,
  where the discrete hard gate often rejects every candidate.
- **`mixed`**: weighted average, `(1 - w) * hard + w * soft`, with `w`
  set by `gate_mixed_weight` (default `0.5`).

Default is `hard`. Use the optional feature config below to switch.

### Optional feature configs

These are **not** default SkillOpt settings — they are optional feature configs
contributed by users for specific scenarios. The paper-reported numbers
were obtained with the default settings, not these.

- **[`configs/features/soft_gate.yaml`](configs/features/soft_gate.yaml)**
  *(PR #25, contributed by [@lvbaocheng](https://github.com/lvbaocheng))* —
  switches `gate_metric` to `soft` (or `mixed`). See the comment at the
  top of the file for when to use and when not to.

---

## Extensibility & WebUI

### Adding a new backend

A backend = a chat / exec target (e.g. `openai_chat`, `claude_chat`,
`qwen_chat`, `minimax_chat`, `codex_exec`, `claude_code_exec`). See
[`docs/guide/new-backend.md`](docs/guide/new-backend.md) for the full
contract; in short you add a `skillopt/model/<name>_backend.py` module,
register it in `skillopt/model/common.py` + `backend_config.py`, and wire
it through the router in `skillopt/model/__init__.py`. `qwen_backend.py`
and `minimax_backend.py` are good templates.

### Adding a new benchmark

A benchmark = a `skillopt/envs/<name>/` package with a `dataloader.py`, a
`rollout.py`, and an `initial.md` seed skill. See
[`docs/guide/new-benchmark.md`](docs/guide/new-benchmark.md) for the full
contract; the simplest reference is `skillopt/envs/searchqa/`.

### WebUI

Launch the monitoring dashboard (optional):

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

| Flag | Default | Description |
|---|---|---|
| `--port` | 7860 | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--share` | off | Create a public Gradio share link |

---

## Citation

```bibtex
@misc{yang2026skilloptexecutivestrategyselfevolving,
      title={SkillOpt: Executive Strategy for Self-Evolving Agent Skills}, 
      author={Yifan Yang and Ziyang Gong and Weiquan Huang and Qihao Yang and Ziwei Zhou and Zisu Huang and Yan Li and Xuemei Gao and Qi Dai and Bei Liu and Kai Qiu and Yuqing Yang and Dongdong Chen and Xue Yang and Chong Luo},
      year={2026},
      eprint={2605.23904},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.23904}
}
```
