# Lambda box quickstart

Everything under `/home/ubuntu` dies when you terminate the instance. Only
`/lambda/nfs/qwen32b` survives. So each new box needs miniconda, the conda env, the code,
and `.env` rebuilt — about **10–15 minutes**, ~$1 of instance time.

The 62 GB of QwQ-32B weights do NOT need re-downloading. They live on the filesystem.

Run `<LOCAL>` blocks on your Mac, `<BOX>` blocks over SSH. The instance IP changes every
launch — grab it from the Lambda console.

---

## 0. Launch — three things that cannot be fixed later

1. SSH key must already be in the Lambda workspace (you're prompted for it at launch).
2. **Attach the `qwen32b` filesystem at creation.** It cannot be attached afterwards.
3. **Same region as the filesystem.** Filesystems don't move between regions.

Instance: **1× GH200 (96 GB)** preferred, or 1× H100 SXM. Ubuntu 22.04 + Lambda Stack.

GH200 is a Grace CPU + H100 on one module. The GPU is *better* than a standalone H100 SXM —
96 GB HBM3 at ~4 TB/s vs 80 GB at 3.35 TB/s — and it costs $2.29/hr vs $4.29. Generation is
bandwidth-bound, so it is also ~20% faster: effective cost per unit of work is under half.

The catch is the **host CPU is ARM64 (aarch64)**, which changes which Python wheels you
install — see the aarch64 fork in step 3. Nothing in the port code is architecture-specific.

Before picking GH200, check the console that **your filesystem's region offers it**. If not
you will be re-downloading 62 GB of weights — fine, but do it knowingly.

---

## 1. `<LOCAL>` Push the code

```bash
export BOX=ubuntu@<INSTANCE-IP>
rsync -av -e "ssh -i ~/Downloads/lambda_labs.pem" \
  --exclude .git --exclude __pycache__ --exclude results \
  ~/Desktop/mats/steering-thinking-llms/ $BOX:~/steering-thinking-llms/
```

**Do this before step 3.** Step 3 does `cd ~/steering-thinking-llms` and `pip install -e .`,
both of which need the repo already on the box. This is the one step that runs on your Mac.

## 2. `<LOCAL>` SSH in

```bash
ssh -i ~/Downloads/lambda_labs.pem ubuntu@<INSTANCE-IP>
```

## 3. `<BOX>` Miniconda + environment

Pick the installer matching `uname -m` — **x86_64**:

```bash
curl -fsSLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash ~/miniconda.sh -b -p ~/miniconda3 \
  && ~/miniconda3/bin/conda init bash \
  && source ~/.bashrc
```

or **aarch64** (GH200):

```bash
curl -fsSLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
  && bash ~/miniconda.sh -b -p ~/miniconda3 \
  && ~/miniconda3/bin/conda init bash \
  && source ~/.bashrc
```

Check which architecture you are on:

```bash
uname -m
```

`x86_64` → H100 SXM path. `aarch64` → GH200 path.

### x86_64 (H100 SXM)

```bash
cd ~/steering-thinking-llms \
  && conda env create -f environment-qwq.yaml \
  && conda activate stllms_env \
  && pip install -e .
```

### aarch64 (GH200)

**torch must come from PyTorch's own index, installed first.** PyPI's aarch64 `torch` wheel
is CPU-only — 92 MB versus 906 MB for the CUDA build. `pip install torch==2.5.1` succeeds,
`torch.cuda.is_available()` is then False, `get_device()` returns CPU, and you run a 32B
model on the Grace CPU with **no error message**. This is the one way to waste hours here.

```bash
cd ~/steering-thinking-llms \
  && conda create -y -n stllms_env python=3.10 \
  && conda activate stllms_env \
  && pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install transformers==4.47.1 nnsight==0.4.5 accelerate==1.2.1 \
  huggingface-hub==0.27.0 safetensors==0.4.5 numpy==1.24.4 \
  openai==1.70.0 anthropic==0.49.0 python-dotenv==1.0.1 \
  matplotlib==3.10.0 seaborn==0.13.2 tqdm==4.67.0 \
  && pip install -e .
```

Installing torch first means the later resolve sees it already satisfied and will not
substitute the PyPI wheel. Everything else has aarch64 cp310 wheels (`safetensors`,
`tokenizers`, `numpy`, `matplotlib` all publish them; `nnsight` is pure Python).

### Both paths

Use `environment-qwq.yaml`, **not** `environment.yaml` — the latter is a dump of the
original author's machine and fails on `ruckig==0.14.0`, which is sdist-only and whose
build config `scikit-build-core >= 0.10` rejects.

**Verify CUDA before anything else** — mandatory on aarch64, cheap on x86_64:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`cuda.is_available()` must be `True`. False on aarch64 means you have the CPU wheel;
reinstall torch from the cu124 index above.

Miniconda goes on the root volume deliberately: a conda env is tens of thousands of small
files, and over NFS every Python import would be slow.

## 4. `<BOX>` Point at the weights

```bash
echo 'export HF_HOME=/lambda/nfs/qwq32b/hf' >> ~/.bashrc && source ~/.bashrc
```

```bash
ls $HF_HOME/hub && du -sh $HF_HOME/hub/models--Qwen--QwQ-32B
```

Must show `models--Qwen--QwQ-32B` at **62G**. If it's missing or small, `HF_HOME` is wrong
or pointing at the root volume — fix it before running anything, or you'll silently
re-download 65 GB.

## 5. `<BOX>` API key

```bash
printf 'OPENAI_API_KEY=sk-YOURREALKEY\n' > ~/steering-thinking-llms/.env
```

Only needed for `train_vectors.py` / `annotate.py`, not for generation.

## 6. `<BOX>` Pre-flight

```bash
cd ~/steering-thinking-llms/train-steering-vectors && python test_qwq_port.py
```

Tokenizer-only, no GPU. Expect `ALL CHECKS PASSED`. Catches the port bugs before you spend
GPU time.

---

## Running work

Always in tmux — a dropped SSH connection kills a bare process (the instance keeps billing
either way).

```bash
tmux new -s run
```

Detach `Ctrl-b` then `d` (Control, not Command; two steps, not a chord). Reattach
`tmux attach -t run`. Or type `tmux detach`.

```bash
cd ~/steering-thinking-llms/train-steering-vectors
python generate_responses.py --model Qwen/QwQ-32B --n_samples 50 --batch_size 4 --max_tokens 2000 --save_every 1 --seed 42
python annotate.py           --model Qwen/QwQ-32B --n_samples 50 --max_workers 8 --seed 42
python train_vectors.py      --model Qwen/QwQ-32B --n_samples 50 --batch_size 2 --save_every 1 --seed 42 --annotate never
```

`--n_samples` and `--seed` **must match across all three** or they select different subsets.

**Batch sizes by GPU.** 65.5 GB of bf16 weights, and the trace holds all 64 layers of
activations at once (~0.65 GB per sequence at 2100 tokens):

| | generation | train_vectors trace |
|---|---|---|
| H100 SXM 80 GB (~14 GB headroom) | `--batch_size 4` | `--batch_size 2` |
| GH200 96 GB (~30 GB headroom) | `--batch_size 8` | `--batch_size 4`–`8` |

GH200's 432 GiB host RAM also means you can leave `--activation_dtype` at float32 rather
than dropping to float16 to save space.

`annotate.py` needs no GPU. For big runs, terminate the box and run it on your laptop.

### Health lines to read, not scroll past

- `QwQ PORT (change 1)` pad token → `<|endoftext|>`
- `QwQ PORT (change 2)` sampling enabled — if missing, greedy decoding will produce
  repetition loops
- `QwQ PORT (change 4)` storing with `skip_special_tokens=False`
- `closed-</think> rate` — under ~50% means raise `--max_tokens`
- `tokenization drift` — should be a token or two near the prompt/completion boundary, not
  scattered through the reasoning
- `emitted` vs `located` span table — a large gap means `get_label_positions` is failing to
  find spans and the counts are undercounts

---

## Before terminating

The root volume goes away. Pull results down:

```bash
rsync -av -e "ssh -i ~/Downloads/lambda_labs.pem" \
  $BOX:~/steering-thinking-llms/train-steering-vectors/results/ \
  ~/Desktop/mats/steering-thinking-llms/train-steering-vectors/results/
```

Anything edited **on the box** is lost — and overwritten by the next push anyway. Edit on the
laptop, treat the box as a destination.

The filesystem bills (~$1/day) while it exists, even unmounted. Delete it when the phase is
done — but only once the weights are no longer needed.

---

## Gotchas, all learned the hard way

| Symptom | Cause |
|---|---|
| `Identity file /home/ubuntu/... not accessible` | You ran rsync **on the box**. It runs from the Mac. |
| `huggingface-cli: deprecated and no longer works` | Use `hf download ...` |
| `Pip failed` / `cmake.targets` | You used `environment.yaml`. Use `environment-qwq.yaml`. |
| `module 'utils' has no attribute '_x'` | `utils/__init__.py` is `from .utils import *`, which skips underscore-prefixed names. |
| Prompt shows `(base)` | `conda activate stllms_env` |
| Re-downloading the model | `HF_HOME` unset or wrong in that shell. It must be the **parent** of `hub/`. |
| `Ctrl-b` doesn't detach | It's Control, not Command — and press `Ctrl-b`, release, *then* `d`. |
| Inexplicably slow, GPU idle (GH200) | PyPI's aarch64 torch is CPU-only. Reinstall from `--index-url https://download.pytorch.org/whl/cu124` and check `torch.cuda.is_available()`. |
