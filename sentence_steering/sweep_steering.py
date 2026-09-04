"""
Does the trained vector actually steer? Layer x coefficient sweep with continuous steering.

This is the Phase 1 go/no-go. validate_vectors.py says whether a direction exists in the
activations; this says whether pushing along it changes the model's behaviour.

Deliberately run on the SAME prompt domain the vector was extracted from (the repo's
eval_messages). A null result here means the vector is bad. Testing on target-domain prompts
before establishing that would confound "vector doesn't work" with "doesn't transfer" -- run
the transfer test separately, after fixing layer and coefficient here.

Conditions per prompt:
  baseline   no steering
  steered    + coefficient * feature[layer], applied at every generated position
  random     + coefficient * random vector of the same norm  (the S-rand control: if this
             moves the metric as much as `steered`, you are measuring noise injection)

Scoring:
  cues    free, instant: count of reconsideration markers. Noisy but enough to see a sweep.
  judge   the real measure: gpt-4.1 labels the output and we count ["backtracking"] spans.
          Costs an API call per generation, so it is opt-in with --judge.
  chars   crude fluency guard. A coefficient that collapses output length is producing
          degenerate text, and any model would "correct" that -- exclude those cells.

  python sweep_steering.py --model Qwen/QwQ-32B --layers 27 29 31 --coefficients 1 2 4 \
      --n_prompts 5 --max_tokens 400 --judge
"""

import argparse
import json
import os
import re

import _paths  # noqa: F401  -- sys.path for utils/messages, vectors dir, .env

import torch
from tqdm import tqdm

import utils
from messages import eval_messages

parser = argparse.ArgumentParser(description="Sweep steering layers and coefficients")
parser.add_argument("--model", type=str, default="Qwen/QwQ-32B")
parser.add_argument("--label", type=str, default="backtracking")
parser.add_argument("--layers", type=int, nargs="+", default=[27, 29, 31])
parser.add_argument("--coefficients", type=float, nargs="+", default=[1.0, 2.0, 4.0])
parser.add_argument("--n_prompts", type=int, default=5)
parser.add_argument("--max_tokens", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--judge", action="store_true", default=False,
                    help="Also score with the annotation judge (one API call per generation)")
parser.add_argument("--judge_model", type=str, default="gpt-4.1")
parser.add_argument("--no_random_control", action="store_true", default=False)
parser.add_argument("--out", type=str, default=str(_paths.RESULTS_DIR / "sweep_steering.json"))
args, _ = parser.parse_known_args()

CUES = re.compile(
    r"\b(wait|hold on|actually|alternatively|hmm|but wait|let me reconsider|"
    r"on second thought|that'?s wrong|scratch that|let me try)\b", re.I)

# Measured on the L29/alpha=2 pilot: cue COUNT anticorrelates with the real effect. Baseline
# says "Wait" 2-4 times during genuine reasoning; a steered response says it once at the top
# and then rambles. The signal is not how often the model reconsiders, it is that it opens in
# reconsideration mode before any reasoning has happened -- referring back to an attempt that
# does not exist. That separated 5/5 from 0/5 where cue count pointed the wrong way.
OPENING_CUE = re.compile(
    r"(wait|again|different approach|made a mistake|reconsider|"
    r"maybe the (question|problem)|actually|alternatively)", re.I)
OPENING_CHARS = 60


def generate(model, tokenizer, message, vector, layer, coefficient):
    """One generation, optionally with a vector added at every position of one layer."""
    input_ids = tokenizer.apply_chat_template(
        [message], add_generation_prompt=True, return_tensors="pt"
    ).to(utils.get_device())

    with torch.no_grad(), model.generate(
        {"input_ids": input_ids,
         "attention_mask": (input_ids != tokenizer.pad_token_id).long()},
        max_new_tokens=args.max_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        model.model.layers.all()
        if vector is not None:
            v = vector.to(utils.get_device(), dtype=utils.get_model_dtype())
            model.model.layers[layer].output[0][:, :] += coefficient * v.unsqueeze(0).unsqueeze(0)
        out = model.generator.output.save()

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # keep only what was generated, not the prompt echo
    marker = "<think>"
    return text[text.index(marker) + len(marker):] if marker in text else text


def ngram_diversity(text, n=6):
    """Unique n-grams / total. Catches semantic looping.

    The alpha=2 pilot degenerated into near-repeats with small variations, which an
    exact-repeat detector misses entirely (max repeated 12-gram was still 1). Diversity
    caught it cleanly: 0.86 steered vs 0.99 baseline / 0.98 random.
    """
    words = text.split()
    if len(words) < n:
        return 1.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def score(text):
    return {
        "chars": len(text),
        "cues": len(CUES.findall(text)),
        "opens_reconsidering": bool(OPENING_CUE.search(text[:OPENING_CHARS])),
        "diversity": round(ngram_diversity(text), 3),
    }


def judge_backtracking(text):
    annotated = utils.annotate_thinking(text, model=args.judge_model)
    if not annotated:
        return None
    return annotated.count(f'["{args.label}"]')


def main():
    print(f"Loading {args.model}...")
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(
        compute_features=True, model_name=args.model)

    if args.label not in feature_vectors:
        raise KeyError(f"No '{args.label}' feature. Run train_vectors.py first.")

    torch.manual_seed(args.seed)
    prompts = eval_messages[:args.n_prompts]
    results = []

    # Conditions: baseline once per prompt, then each (layer, coefficient), plus the control.
    cells = [("baseline", None, None)]
    for layer in args.layers:
        for coefficient in args.coefficients:
            cells.append(("steered", layer, coefficient))
            if not args.no_random_control:
                cells.append(("random", layer, coefficient))

    for name, layer, coefficient in tqdm(cells, desc="cells"):
        for i, message in enumerate(prompts):
            if name == "baseline":
                vector = None
            elif name == "steered":
                vector = feature_vectors[args.label][layer]
            else:
                # norm-matched random direction: isolates "did adding *any* vector of this
                # size do it" from "did adding *this* vector do it"
                reference = feature_vectors[args.label][layer]
                vector = torch.randn_like(reference)
                vector = vector * (reference.norm() / vector.norm())

            text = generate(model, tokenizer, message, vector, layer, coefficient)
            row = {"condition": name, "layer": layer, "coefficient": coefficient,
                   "prompt_idx": i, "text": text, **score(text)}
            if args.judge:
                row["judged_backtracking"] = judge_backtracking(text)
            results.append(row)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # --- report ---------------------------------------------------------------
    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    base = [r for r in results if r["condition"] == "baseline"]
    base_chars, base_div = agg(base, "chars"), agg(base, "diversity")

    def line(name, layer, coefficient, rows):
        opens = sum(1 for r in rows if r["opens_reconsidering"])
        # Two-sided length guard: degeneration can inflate length (never terminating) as
        # easily as collapse it. The alpha=2 pilot ran 60% LONG, which a one-sided guard missed.
        ratio = agg(rows, "chars") / base_chars if base_chars else 1.0
        div = agg(rows, "diversity")
        flag = ""
        if div < base_div - 0.05:
            flag = "  <-- DEGENERATE (low n-gram diversity)"
        elif not 0.5 < ratio < 1.5:
            flag = f"  <-- length {ratio:.1f}x baseline"
        print(f"{name:10s} {str(layer):>5s} {str(coefficient):>5s} "
              f"{opens:>3d}/{len(rows):<3d} {agg(rows,'cues'):6.1f} {agg(rows,'chars'):7.0f} "
              f"{div:9.3f}"
              + (f"{agg(rows,'judged_backtracking'):8.1f}" if args.judge else "") + flag)

    judged = "judged" if args.judge else ""
    print(f"\n{'condition':10s} {'layer':>5s} {'coef':>5s} {'opens':>7s} {'cues':>6s} "
          f"{'chars':>7s} {'diversity':>9s} {judged:>8s}")
    print("  (opens = responses starting in reconsideration mode -- the headline metric)")
    line("baseline", "-", "-", base)
    for layer in args.layers:
        for coefficient in args.coefficients:
            for name in ("steered", "random"):
                rows = [r for r in results if r["condition"] == name
                        and r["layer"] == layer and r["coefficient"] == coefficient]
                if rows:
                    line(name, layer, coefficient, rows)

    print(f"\nwrote {args.out}")
    print("\nRead the actual text before believing the numbers. A cell counts as usable only "
          "if `opens` clearly exceeds both `baseline` and `random` AND diversity stays near "
          "baseline. At L29/alpha=2 the pilot hit 5/5 opens with diversity 0.86 vs 0.99 -- "
          "the behaviour was induced and the model then broke. You want the largest "
          "coefficient that keeps diversity intact.")


if __name__ == "__main__":
    main()
