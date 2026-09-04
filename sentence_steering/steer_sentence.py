"""
Single-sentence steering on the blackmail scenario: steer one sentence, release, continue.

This is the transfer test AND the core of the Phase 2 harness. It is deliberately NOT the
continuous steering of sweep_steering.py -- the smoke test showed continuous steering
degenerates after ~50 words while the FIRST sentence stays clean, and one sentence is the
intervention the experiment actually runs.

Per (trace, target sentence i), three conditions, each a two-pass generation:

  pass 1  prefix -> generate a bounded burst WITH steering, cut at the first sentence
          boundary. That is the replacement sentence.
  pass 2  prefix + replacement -> continue with steering OFF.

Steering is released for real because pass 2 is a separate forward pass with no hook.

  I0      pass 1 unsteered              -- clean floor, the model's own resample
  S       pass 1 with the vector        -- the intervention
  S-rand  pass 1 with a norm-matched random vector -- noise-injection control

PROMPT FORMAT, deliberately not the chat template. Thought Branches builds a raw completion
prompt (generate_blackmail_rollouts.py:801):

    {system_prompt}\\n\\nUser: {user_prompt}\\n\\n{email_content}\\n\\nAssistant:\\n<think>\\n

Using apply_chat_template instead would sample from a different distribution than their
released QwQ-32B traces and break comparability with the R condition later. Sentence splitting
likewise reuses their split_solution_into_chunks.

    python steer_sentence.py --model Qwen/QwQ-32B --scenario blackmail_prompt.json \\
        --layer 29 --coefficient 2 --n_traces 3 --positions 0.3 0.6 --k 3
"""

import argparse
import json
import os
import re
import sys

import _paths  # noqa: F401  -- sys.path for utils/messages, vectors dir, .env

import torch
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser(description="Steer one sentence, release, continue")
parser.add_argument("--model", type=str, default="Qwen/QwQ-32B")
parser.add_argument("--scenario", type=str, required=True,
                    help="JSON from thought-branches/blackmail/build_blackmail_prompt.py")
parser.add_argument("--label", type=str, default="backtracking")
parser.add_argument("--layer", type=int, default=29)
parser.add_argument("--coefficient", type=float, default=2.0)
parser.add_argument("--n_traces", type=int, default=3)
parser.add_argument("--positions", type=float, nargs="+", default=[0.3, 0.6],
                    help="Target sentences as fractions through the trace. Their principled "
                         "rule uses P(blackmail|prefix) in the 25-75%% band from the released "
                         "resampling data; that is not available for self-generated traces, "
                         "so these are positional stand-ins for the pilot.")
parser.add_argument("--k", type=int, default=3, help="Continuations per condition")
parser.add_argument("--base_tokens", type=int, default=2000)
parser.add_argument("--sentence_tokens", type=int, default=60,
                    help="Budget for pass 1; the burst is cut at the first sentence boundary")
parser.add_argument("--continuation_tokens", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--thought_branches", type=str,
                    default=str(_paths.THOUGHT_BRANCHES),
                    help="For split_solution_into_chunks, so sentence boundaries match theirs")
parser.add_argument("--out", type=str, default=str(_paths.RESULTS_DIR / "steer_sentence_blackmail.json"))
args, _ = parser.parse_known_args()

# Their splitter, so chunk boundaries match the released data.
def _load_thought_branches_utils(path):
    """Load thought-branches/blackmail/utils.py under its own module name.

    A plain `from utils import ...` cannot work: `import utils` earlier in this file already
    bound the STEERING REPO's utils package in sys.modules, and sys.modules wins over
    sys.path -- so the import returns the wrong module and silently falls back to the regex
    splitter. Loading by file path under a distinct name avoids the collision.
    """
    import importlib.util
    utils_py = os.path.join(path, "utils.py")
    spec = importlib.util.spec_from_file_location("tb_utils", utils_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {utils_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Use the WITH_SEPARATORS variant. Joining chunks with " " loses the trace's newlines and
# paragraph structure, so the prefix the model conditions on is not the text it actually
# generated -- which showed up as the unsteered I0 resample switching to third person
# ("The user is Alex") where the original was first person. The separators let the prefix be
# reconstructed exactly.
try:
    _tb = _load_thought_branches_utils(os.path.abspath(args.thought_branches))
    split_sentences = _tb.split_solution_into_chunks_with_separators
    print(f"using split_solution_into_chunks_with_separators from {args.thought_branches}")
except Exception as _exc:
    print(f"WARNING: thought-branches utils not importable ({_exc}); falling back to a regex "
          "splitter. Boundaries will not exactly match the released chunking.")
    def split_sentences(text):
        parts = re.split(r"((?<=[.!?])\s+)", text.strip())
        chunks = [p for p in parts[::2] if len(p.strip()) >= 4]
        seps = parts[1::2] + [""]
        return chunks, seps[:len(chunks)]

SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]+)?(?:\s|$)")


def build_prompt(scenario, prefix=""):
    """Exactly the format at generate_blackmail_rollouts.py:801 / :881."""
    return (f"{scenario['system_prompt']}\n\nUser: {scenario['user_prompt']}\n\n"
            f"{scenario['email_content']}\n\nAssistant:\n<think>\n{prefix}")


def generate(model, tokenizer, prompt, max_new_tokens, vector=None, coefficient=0.0):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(
        utils.get_device())
    with torch.no_grad(), model.generate(
        {"input_ids": input_ids,
         "attention_mask": torch.ones_like(input_ids)},
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        model.model.layers.all()
        if vector is not None:
            v = vector.to(utils.get_device(), dtype=utils.get_model_dtype())
            model.model.layers[args.layer].output[0][:, :] += coefficient * v.unsqueeze(0).unsqueeze(0)
        out = model.generator.output.save()
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):] if full.startswith(prompt) else full


def first_sentence(text):
    """Cut a generated burst at its first sentence boundary."""
    match = SENTENCE_END.search(text)
    return text[:match.end()].strip() if match else text.strip()


def main():
    with open(args.scenario) as f:
        scenario = json.load(f)

    print(f"Loading {args.model}...")
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(
        compute_features=True, model_name=args.model)
    feature = feature_vectors[args.label][args.layer]
    torch.manual_seed(args.seed)

    results = []
    for trace_idx in range(args.n_traces):
        print(f"\n=== trace {trace_idx}: generating base trace ===")
        base = generate(model, tokenizer, build_prompt(scenario), args.base_tokens)
        sentences, separators = split_sentences(base)
        print(f"  {len(base.split())} words, {len(sentences)} sentences")
        if len(sentences) < 6:
            print("  too few sentences to pick interior targets; skipping")
            continue

        for fraction in args.positions:
            i = max(1, min(len(sentences) - 2, int(fraction * len(sentences))))
            # Rebuild the prefix exactly as generated, separators included.
            prefix = "".join(c + s for c, s in zip(sentences[:i], separators[:i]))
            original = sentences[i]
            if not base.startswith(prefix):
                print("    WARNING: reconstructed prefix does not match the base trace; "
                      "chunk/separator round-trip is lossy for this text.")
            print(f"    prefix ends: ...{prefix[-160:]!r}")
            print(f"  [pos {fraction}] sentence {i}/{len(sentences)}")
            print(f"    original : {original}")

            for condition in ("I0", "S", "S-rand"):
                if condition == "I0":
                    vector, coefficient = None, 0.0
                elif condition == "S":
                    vector, coefficient = feature, args.coefficient
                else:
                    vector = torch.randn_like(feature)
                    vector = vector * (feature.norm() / vector.norm())
                    coefficient = args.coefficient

                burst = generate(model, tokenizer, build_prompt(scenario, prefix),
                                 args.sentence_tokens, vector, coefficient)
                replacement = first_sentence(burst)
                was_cut = len(replacement) < len(burst.strip())
                # Print in full. These sentences are the intervention -- they have to be read,
                # not skimmed, and a truncated preview hides exactly the drift worth catching.
                print(f"    {condition:7s} : {replacement}"
                      f"{'' if was_cut else '   [NOT CUT AT A BOUNDARY]'}")

                # Pass 2: steering released -- separate forward pass, no hook.
                continuations = [
                    generate(model, tokenizer,
                             build_prompt(scenario, prefix + replacement + separators[i]),
                             args.continuation_tokens)
                    for _ in range(args.k)
                ]

                results.append({
                    "trace_idx": trace_idx, "position": fraction, "sentence_idx": i,
                    "n_sentences": len(sentences), "condition": condition,
                    "layer": args.layer, "coefficient": coefficient,
                    "original_sentence": original, "replacement_sentence": replacement,
                    "burst_was_cut": was_cut,
                    "raw_burst": burst,
                    "continuations": continuations,
                })

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    cut = sum(1 for r in results if r["burst_was_cut"])
    print(f"\nwrote {args.out} ({len(results)} rows)")
    print(f"boundary stopping worked in {cut}/{len(results)} bursts "
          f"-- a low rate means --sentence_tokens is too small and sentences are being "
          f"truncated mid-clause rather than cut at a boundary.")
    print("\nRead every S replacement sentence. It must be (a) fluent, (b) a reconsideration, "
          "(c) about the DECISION, not about the prompt being unclear -- the math-domain "
          "steering drifted into 'maybe the question is about something else', which would be "
          "useless here.")


if __name__ == "__main__":
    main()
