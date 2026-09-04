# sentence_steering

Project code for the sentence-level steering experiment. The steering **vectors** are trained
inside the forked repo (`../steering-thinking-llms`); everything downstream lives here.

## Layout

```
mats/
  sentence_steering/          this package -- all project code
    _paths.py                 central paths; every script imports it first
    fetch_blackmail_rollouts.py   pull QwQ rollouts from HF into ../data/blackmail (duckdb, no GPU)
    build_blackmail_prompt.py     assemble the Lynch et al. prompt (no GPU)
    validate_vectors.py           inspect a trained mean_vectors_*.pt (no GPU)
    sweep_steering.py             layer x coefficient continuous-steering sweep (GPU)
    steer_sentence.py             single-sentence steer/release pilot (GPU)
    phase2_generate.py            Phase 2: S / S-rand on released prefixes (GPU)
  data/blackmail/             fetched rollouts, shard_manifest.json, targets.json
  results/vars/               script outputs (sweeps, generations)
  steering-thinking-llms/     forked repo: vector-training pipeline + utils/messages
    train-steering-vectors/   train_vectors.py, generate_responses.py, annotate.py, the .pt files
```

## What stays in the fork vs here

`steering-thinking-llms` keeps the vector-generation pipeline (`train_vectors.py`,
`generate_responses.py`, `annotate.py`, `test_qwq_port.py`) and the shared `utils` / `messages`
packages. This directory holds all code written for the experiment itself.

## How the wiring works (`_paths.py`)

Each script does `import _paths` before importing `utils`. That:
- puts `steering-thinking-llms` on `sys.path`, so `import utils` / `import messages` resolve
  without needing `pip install -e .` or a particular cwd;
- sets `MEAN_VECTORS_DIR` so `utils.load_model_and_vectors` finds the trained `.pt` files at
  their absolute location;
- loads `mats/.env` (OPENAI_API_KEY) if python-dotenv is installed.

Run scripts directly, from anywhere:

    python sentence_steering/fetch_blackmail_rollouts.py index
    python sentence_steering/validate_vectors.py --model Qwen/QwQ-32B
    python sentence_steering/phase2_generate.py --model Qwen/QwQ-32B --layer 29 --coefficient 1

GPU scripts need the `stllms_env` conda env; `fetch_*` needs `duckdb` (`pip install duckdb`).

## Data note

`data/blackmail/` currently holds `shard_manifest.json` (all 20 QwQ scenarios) plus a
scenario_0 sample pulled while testing the move. Run `pull-small` (no `--limit`) then
`targets` / `pull-chunks` to populate the full target set.
