"""
Phase 2 generation: produce S and S-rand on the RELEASED blackmail prefixes.

I0 and R are extractions from the dataset (chunk_{i+1} and chunk_i solutions.json), so this
script generates only S and S-rand. Crucially it builds them on the SAME prefix the released
rollouts used -- chunk_i/solutions.json[*]["prefix_without_chunk"], the reasoning up to (not
including) sentence i, identical across all 100 rollouts at that chunk. That is what makes the
S-vs-R / S-vs-I0 comparison paired at (trace, position).

Per target (scenario, chunk_idx i), for each condition:
  pass 1  build_prompt(prefix)                 -> steered burst, cut at first sentence -> replacement
  pass 2  build_prompt(prefix + replacement)   -> k continuations, steering OFF (separate pass)

Prompt format is Thought Branches' raw completion prompt (generate_blackmail_rollouts.py:801),
NOT the chat template -- matching how the released rollouts were generated.

Inputs come from fetch_blackmail_rollouts.py:
  <data_dir>/qwq-32b/.../scenario_<id>/scenario.json
  <data_dir>/qwq-32b/.../scenario_<id>/chunk_<i>/solutions.json   (for prefix_without_chunk)
  <data_dir>/targets.json

    python phase2_generate.py --model Qwen/QwQ-32B --data_dir data/blackmail \\
        --layer 29 --coefficient 1 --k 10 --conditions S S-rand
"""

import argparse
import hashlib
import json
import os
import random
import sys

import _paths  # noqa: F401  -- sys.path for utils/messages, vectors dir, .env

import torch
from tqdm import tqdm

import utils

QWQ_PREFIX = "qwq-32b/temperature_0.7_top_p_0.95/yes_base_solution"

parser = argparse.ArgumentParser(description="Generate S and S-rand on released blackmail prefixes")
parser.add_argument("--model", type=str, default="Qwen/QwQ-32B")
parser.add_argument("--data_dir", type=str, default=str(_paths.DATA_DIR))
parser.add_argument("--targets", type=str, default=None, help="defaults to <data_dir>/targets.json")
parser.add_argument("--label", type=str, default="backtracking")
parser.add_argument("--layer", type=int, default=29)
parser.add_argument("--coefficient", type=float, default=1.0)
parser.add_argument("--conditions", type=str, nargs="+", default=["S", "E"],
                    choices=["S", "S-rand", "E"],
                    help="I0 and R are dataset extractions, not generated here. E takes its "
                         "replacement sentence from --edits (written offline by "
                         "generate_edits.py) rather than generating a burst.")
parser.add_argument("--edits", type=str, default=None,
                    help="edits.json from generate_edits.py; required when 'E' is in "
                         "--conditions. Defaults to <data_dir>/edits.json")
parser.add_argument("--k", type=int, default=10,
                    help="continuations per burst (samples continuation variance)")
parser.add_argument("--steer_scope", type=str, default="generated",
                    choices=["generated", "all"],
                    help="WHERE the steering vector is added. 'generated' (default, and what "
                         "this experiment's design calls for): only the positions being "
                         "generated. 'all': every position including the ~3.5k prompt tokens "
                         "-- the paper's continuous whole-context intervention, kept for A/B "
                         "comparison. See generate_batch for why this matters.")
parser.add_argument("--n_bursts", type=int, default=1,
                    help="Independent draws of the STEERED SENTENCE itself per target. "
                         "R's per-target estimate averages over ~100 different replacement "
                         "sentences; with n_bursts=1, S's rests on a single draw, so the two "
                         "are not comparable and there is nothing to subsample for the "
                         "matched-strength step. It also makes intervention reliability "
                         "(H-fragile) observable. Total generations per target/condition is "
                         "n_bursts * (1 + k), so trade against --k rather than adding to it.")
parser.add_argument("--tag", type=str, nargs="+", default=None,
                    help="Only process targets whose function_tags include one of these "
                         "(e.g. --tag plan_generation leverage_identification). Phase 1 "
                         "suggested steering lands on strategy sentences but not "
                         "situation_assessment ones; this makes that split testable.")
parser.add_argument("--skip_missing", action="store_true", default=False,
                    help="Proceed with only the targets whose solutions.json is present, "
                         "instead of exiting so you can run pull-chunks.")
parser.add_argument("--limit", type=int, default=None,
                    help="Only process the first N targets. Use for smoke tests before "
                         "committing to the full target set.")
parser.add_argument("--cudnn_sdpa", action="store_true", default=False,
                    help="Re-enable the cuDNN attention backend. Off by default: it "
                         "JIT-compiles shape-specialised kernels, and our left-padded "
                         "continuation batches keep changing shape, so it allocates workspace "
                         "OUTSIDE PyTorch's allocator and dies with "
                         "CUDNN_STATUS_EXECUTION_FAILED once the allocator holds most of VRAM.")
parser.add_argument("--memory_fraction", type=float, default=0.92,
                    help="Cap PyTorch's share of VRAM, leaving headroom for attention/NCCL "
                         "scratch that lives outside its allocator. 1.0 disables the cap.")
parser.add_argument("--max_batch", type=int, default=8,
                    help="Rows per batched generate call. The binding constraint is PREFILL "
                         "activation, not the KV cache: prefill materialises a "
                         "batch x seq_len x 27648 MLP intermediate (SwiGLU needs three live "
                         "at once), so at ~3.4k-token prompts that is ~1.5GB per tensor per "
                         "8 rows. Measured: batch 20 OOMs on a 96GB GH200 (65.5GB weights + "
                         "~20GB KV + ~10GB prefill transient). 8 leaves comfortable headroom.")
parser.add_argument("--sentence_tokens", type=int, default=60)
parser.add_argument("--continuation_tokens", type=int, default=3000,
                    help="Upper bound only. Continuations stop as soon as the model closes "
                         "</tool_use:email> -- exactly where the released rollouts terminate "
                         "-- so this is a safety cap, not the usual length. It must be "
                         "generous: their rollouts run to ~2129 words, and a continuation that "
                         "stops before the email is sent scores 'no blackmail' by construction "
                         "under their compound verdict.")
parser.add_argument("--stop_strings", type=str, nargs="*",
                    default=["</tool_use:email>"],
                    help="Stop generation at these strings. ONLY </tool_use:email>: it is the "
                         "true terminator. </SCRATCHPAD_REASONING> looks like one but is NOT -- "
                         "in the released rollouts it is followed by more text 69% of the "
                         "time, and that text is the email itself. Stopping there truncates "
                         "the continuation one step before the action and scores a decided "
                         "blackmail as 'no blackmail'. Pass with no values to disable if the "
                         "transformers/nnsight version does not forward the kwarg.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--thought_branches", type=str, default=str(_paths.THOUGHT_BRANCHES))
parser.add_argument("--out", type=str, default=str(_paths.RESULTS_DIR / "phase2_S.json"))
args, _ = parser.parse_known_args()

# Their sentence splitter, so the S replacement is chunked the way R's chunk_resampled was.

def _load_thought_branches_utils(path):
    """Load thought-branches/blackmail/utils.py under its own module name.

    A plain `from utils import ...` cannot work here: `import utils` earlier in this file
    already bound the STEERING REPO's utils package in sys.modules, and sys.modules wins over
    sys.path -- so the later import returns the wrong module and silently falls back to a
    regex splitter. That matters: the replacement sentence would then be chunked differently
    from R's chunk_resampled, which is exactly the comparability the docstring relies on.
    Loading by file path under a distinct name avoids the collision entirely.
    """
    import importlib.util
    utils_py = os.path.join(path, "utils.py")
    spec = importlib.util.spec_from_file_location("tb_utils", utils_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {utils_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    _tb = _load_thought_branches_utils(os.path.abspath(args.thought_branches))
    split_sentences = _tb.split_solution_into_chunks
    print(f"using split_solution_into_chunks from {args.thought_branches}")
except Exception as _exc:
    import re
    print(f"WARNING: thought-branches splitter unavailable ({_exc}); using regex fallback. "
          f"Replacement sentences will be chunked differently from R's chunk_resampled.")
    def split_sentences(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) >= 4]


def scenario_dir(sid):
    return os.path.join(args.data_dir, QWQ_PREFIX, f"scenario_{sid}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_prompt(scenario, prefix):
    return (f"{scenario['system_prompt']}\n\nUser: {scenario['user_prompt']}\n\n"
            f"{scenario['email_content']}\n\nAssistant:\n<think>\n{prefix}")


def get_prefix(sid, i):
    """The reasoning prefix before sentence i, exactly as the released rollouts used it.

    All 100 rollouts at a chunk share prefix_without_chunk; assert that so a surprise does not
    silently desync S from R.
    """
    sols = load_json(os.path.join(scenario_dir(sid), f"chunk_{i}", "solutions.json"))
    prefixes = {r.get("prefix_without_chunk") for r in sols if r.get("prefix_without_chunk")}
    if len(prefixes) != 1:
        raise ValueError(f"scenario {sid} chunk {i}: {len(prefixes)} distinct prefixes; expected 1")
    return prefixes.pop()


def stop_kwargs(tokenizer):
    """Stop each sequence when it finishes, instead of always running to the cap.

    The released rollouts end exactly at </tool_use:email> (or </SCRATCHPAD_REASONING>), so
    stopping there matches their protocol and avoids paying for thousands of dead tokens after
    the decision is made. HF marks each sequence done independently and pads it, so this is
    batch-safe. Disabled by passing --stop_strings with no values.
    """
    if not args.stop_strings:
        return {}
    return {"stop_strings": list(args.stop_strings), "tokenizer": tokenizer}


def generate_batch(model, tokenizer, prompt, n, max_new_tokens, vectors=None):
    """Generate n samples from ONE prompt in a single batched call.

    Why batch: decoding a 32B model is memory-bandwidth bound -- every decode step reads all
    65.5GB of weights regardless of batch size. Generating one sequence at a time pays a full
    weight read per token. At batch n the same read yields n tokens, so throughput is close to
    linear in n until the KV cache runs out of room (~0.9GB per sequence at 3.5k context).

    Every row shares the same prompt here (the bursts all start from the same prefix; the k
    continuations all start from prefix+replacement), so no left-padding and no per-row
    attention masking is needed -- which also sidesteps the pad-token pitfalls this model has.

    `vectors` is an optional list of length n giving each row its own steering vector (None =
    unsteered). They are stacked into one (n, hidden) tensor and added with a single broadcast
    over the sequence dimension, rather than n separate indexed writes into the intervention
    graph. That is what lets S and S-rand -- and several independent bursts -- share one call.
    """
    # Split into chunks so --n_bursts / --k cannot silently exceed VRAM.
    if n > args.max_batch:
        out = []
        for start in range(0, n, args.max_batch):
            size = min(args.max_batch, n - start)
            sub = None if vectors is None else vectors[start:start + size]
            out.extend(generate_batch(model, tokenizer, prompt, size, max_new_tokens, sub))
        return out

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(
        utils.get_device())
    input_ids = input_ids.repeat(n, 1)

    stacked = None
    if vectors is not None and any(v is not None for v in vectors):
        hidden = next(v for v in vectors if v is not None)
        stacked = torch.stack([
            (v if v is not None else torch.zeros_like(hidden)) for v in vectors
        ]).to(utils.get_device(), dtype=utils.get_model_dtype())   # (n, hidden)

    with torch.no_grad(), model.generate(
        {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        **stop_kwargs(tokenizer),
    ) as tracer:
        if stacked is None:
            model.model.layers.all()
        elif args.steer_scope == "all":
            # Original behaviour: every position, every pass. On the PREFILL pass the layer
            # output is [batch, ~3500, hidden], so this adds the vector to every token of the
            # system prompt, the emails and the reasoning prefix -- steering the model's
            # representation of the whole scenario, not just the sentence being written. That
            # is the paper's continuous intervention, not "steer one sentence then release",
            # and it is the likely cause of the scenario-grounding collapse at alpha=2.
            model.model.layers.all()
            model.model.layers[args.layer].output[0][:, :] += stacked.unsqueeze(1)
        else:
            # 'generated': steer only what is being produced.
            #
            # nnsight's .all() is iter[:] -- every iteration, prefill included. Iteration 0 IS
            # the prefill pass, and it also produces the FIRST generated token (from the final
            # prompt position). So skipping iteration 0 outright would leave that token
            # unsteered. Instead:
            #   iter[0]  -> add at the last position only (the one that emits token 1),
            #               leaving the other ~3.5k prompt positions untouched
            #   iter[1:] -> each decode step sees exactly one new position, so [:, :] is
            #               already scoped to generated text
            with model.model.layers.iter[0]:
                model.model.layers[args.layer].output[0][:, -1:] += stacked.unsqueeze(1)
            with model.model.layers.iter[1:]:
                model.model.layers[args.layer].output[0][:, :] += stacked.unsqueeze(1)
        out = model.generator.output.save()

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t[len(prompt):] if t.startswith(prompt) else t for t in texts]


def first_sentence(burst):
    chunks = split_sentences(burst)
    return chunks[0].strip() if chunks else burst.strip()


def generate_batch_prompts(model, tokenizer, prompts, max_new_tokens, vectors=None):
    """Generate one sample per prompt, where prompts may DIFFER in length.

    generate_batch requires every row to share a prompt, so the k continuations of each burst
    batch together but the bursts do not -- leaving one call per burst. Since continuations
    are 400 decode steps against the burst's 60, those calls are ~97% of the run: at 3 bursts
    x 2 conditions that is 6 x 400 = 2400 sequential steps per target, when all 24
    continuations could share a single 400-step call.

    Differing lengths need left-padding. Two things make that safe here:
      - the tokenizer pads with <|endoftext|>, never <|im_end|> (see utils.select_pad_token),
        so the attention mask cannot swallow real content;
      - padding on the LEFT means every row's final position is real prompt text, which is
        what steer_scope='generated' targets with output[0][:, -1:].
    """
    if not prompts:
        return []
    if len(prompts) > args.max_batch:
        out = []
        for start in range(0, len(prompts), args.max_batch):
            sub_v = None if vectors is None else vectors[start:start + args.max_batch]
            out.extend(generate_batch_prompts(
                model, tokenizer, prompts[start:start + args.max_batch],
                max_new_tokens, sub_v))
        return out

    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(utils.get_device())
    attention_mask = enc["attention_mask"].to(utils.get_device())

    stacked = None
    if vectors is not None and any(v is not None for v in vectors):
        hidden = next(v for v in vectors if v is not None)
        stacked = torch.stack([
            (v if v is not None else torch.zeros_like(hidden)) for v in vectors
        ]).to(utils.get_device(), dtype=utils.get_model_dtype())

    with torch.no_grad(), model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        **stop_kwargs(tokenizer),
    ) as tracer:
        if stacked is None:
            model.model.layers.all()
        elif args.steer_scope == "all":
            model.model.layers.all()
            model.model.layers[args.layer].output[0][:, :] += stacked.unsqueeze(1)
        else:
            with model.model.layers.iter[0]:
                model.model.layers[args.layer].output[0][:, -1:] += stacked.unsqueeze(1)
            with model.model.layers.iter[1:]:
                model.model.layers[args.layer].output[0][:, :] += stacked.unsqueeze(1)
        out = model.generator.output.save()

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t[len(pr):] if t.startswith(pr) else t for t, pr in zip(texts, prompts)]


def main():
    targets_path = args.targets or os.path.join(args.data_dir, "targets.json")
    targets = load_json(targets_path)

    if args.tag:
        wanted = {t.lower() for t in args.tag}

        def tags_of(t):
            tags = t.get("function_tags")
            if tags is None:   # older targets.json without the field
                cl = load_json(os.path.join(scenario_dir(t["scenario_id"]), "chunks_labeled.json"))
                by = {c.get("chunk_idx", j): c for j, c in enumerate(cl)}
                tags = (by.get(t["chunk_idx"], {}) or {}).get("function_tags") or []
            return {str(x).lower() for x in tags}

        before = len(targets)
        targets = [t for t in targets if tags_of(t) & wanted]
        print(f"--tag {' '.join(args.tag)}: {len(targets)}/{before} targets match")
        if not targets:
            sys.exit("No targets match the requested tags.")

    if args.limit:
        targets = targets[:args.limit]
        print(f"{len(targets)} targets (limited from {targets_path})")
    else:
        print(f"{len(targets)} targets from {targets_path}")

    # Preflight: every target needs chunk_<i>/solutions.json for its prefix. Re-running
    # `targets` changes which chunks are selected, so the matching pull-chunks must be re-run
    # too -- otherwise targets are skipped one by one mid-run and the output looks like a
    # steering failure when it is really missing data.
    missing = [t for t in targets
               if not os.path.exists(os.path.join(
                   scenario_dir(t["scenario_id"]), f"chunk_{t['chunk_idx']}", "solutions.json"))]
    if missing:
        print(f"\n{len(missing)}/{len(targets)} targets have no solutions.json (no prefix).")
        for t in missing[:5]:
            print(f"  scenario {t['scenario_id']} chunk {t['chunk_idx']}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        print("\nFetch them first:\n"
              "  python fetch_blackmail_rollouts.py pull-chunks\n"
              "Or pass --skip_missing to proceed with only the targets that have prefixes.")
        if not args.skip_missing:
            sys.exit(1)
        targets = [t for t in targets if t not in missing]
        print(f"--skip_missing: proceeding with {len(targets)} targets\n")

    # --- E arm: load the offline-written edit sentences -------------------------
    edits_by_target = {}
    if "E" in args.conditions:
        edits_path = args.edits or os.path.join(args.data_dir, "edits.json")
        if not os.path.exists(edits_path):
            sys.exit(f"--conditions includes E but {edits_path} not found. "
                     f"Run: python generate_edits.py --tag plan_generation")
        for row in load_json(edits_path):
            edits_by_target[(row["scenario_id"], row["chunk_idx"])] = row["edits"]
        missing = [t for t in targets
                   if (t["scenario_id"], t["chunk_idx"]) not in edits_by_target]
        print(f"E arm: edits for {len(edits_by_target)} targets"
              + (f"; {len(missing)} selected targets have none" if missing else ""))

    # --- VRAM: leave room for allocations outside PyTorch's allocator -----------
    # Attention backends need scratch PyTorch does not manage. Two guards: disable the cuDNN
    # SDPA backend (it JIT-compiles per shape, and left-padded batches keep changing shape),
    # and cap the allocator so that scratch has somewhere to live.
    if not args.cudnn_sdpa and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("cuDNN SDPA disabled (flash / mem-efficient attention instead)")
    if args.memory_fraction < 1.0 and torch.cuda.is_available():
        torch.cuda.memory.set_per_process_memory_fraction(args.memory_fraction)
        print(f"PyTorch allocator capped at {args.memory_fraction:.0%} of VRAM")

    print(f"Loading {args.model}...")
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(
        compute_features=True, model_name=args.model)
    feature = feature_vectors[args.label][args.layer]
    torch.manual_seed(args.seed)

    results = []
    for t in tqdm(targets, desc="targets"):
        sid, i = t["scenario_id"], t["chunk_idx"]
        scenario = load_json(os.path.join(scenario_dir(sid), "scenario.json"))
        chunks_data = load_json(os.path.join(scenario_dir(sid), "chunks.json"))
        original = chunks_data["chunks"][i] if i < len(chunks_data["chunks"]) else ""
        seps = chunks_data.get("separators") or []
        sep = seps[i] if i < len(seps) else " "

        try:
            prefix = get_prefix(sid, i)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  scenario {sid} chunk {i}: skipped ({exc})")
            continue
        # Guard the pairing: S/S-rand/R all share exactly this prefix.
        prefix_hash = hashlib.sha1(prefix.encode()).hexdigest()[:12]

        # --- Stage 2: every burst, for every condition, in ONE batched call -------
        # All of them start from the same prefix prompt and differ only by steering vector,
        # so they are rows of a single generate. S-rand draws a fresh random direction per
        # burst, so its bursts are genuinely independent rather than one vector reused.
        burst_prompt = build_prompt(scenario, prefix)
        # E needs no burst: its sentence was written offline. Only the steered arms are
        # generated here, so E costs continuations only.
        rows, vectors = [], []
        for condition in args.conditions:
            if condition == "E":
                continue
            for burst_idx in range(args.n_bursts):
                if condition == "S":
                    vector = feature
                else:  # norm-matched random direction, redrawn per burst
                    vector = torch.randn_like(feature)
                    vector = vector * (feature.norm() / vector.norm())
                rows.append((condition, burst_idx))
                vectors.append(args.coefficient * vector)

        bursts = (generate_batch(model, tokenizer, burst_prompt, len(rows),
                                 args.sentence_tokens, vectors) if rows else [])

        # Splice the pre-written edits in as if they were bursts, so everything downstream --
        # the single batched continuation call, the per-row records, the aggregation -- treats
        # S and E identically. `burst` is the raw text and equals the replacement for E.
        if "E" in args.conditions:
            available = edits_by_target.get((sid, i), [])
            if not available:
                print(f"  scenario {sid} chunk {i}: no edits found; E skipped for this target")
            # Sample rather than take the first n: if fewer bursts than edits are
            # requested, the subset should be a random draw from the pool, not a fixed
            # prefix of it. Seeded, so a rerun reproduces the same choice.
            chosen = (available if len(available) <= args.n_bursts
                      else random.Random(args.seed).sample(available, args.n_bursts))
            for burst_idx, edit in enumerate(chosen):
                rows.append(("E", burst_idx))
                bursts.append(edit)

        # --- Stage 1: the k continuations for each burst in ONE batched call ------
        # These cannot join the batch above: each row continues from its own replacement
        # sentence, so the prompts differ. Batching across bursts would need left-padding.
        # --- Stage 3: every continuation for this target in ONE left-padded batch -----
        # Previously one call per burst (6 x 400 decode steps); now all n_bursts x
        # n_conditions x k continuations share a single call of 400 steps.
        cont_prompts, meta = [], []
        for (condition, burst_idx), burst in zip(rows, bursts):
            # E is already a single written sentence; cutting it would silently truncate.
            replacement = burst.strip() if condition == "E" else first_sentence(burst)
            was_cut = len(replacement) < len(burst.strip())
            prompt = build_prompt(scenario, prefix + replacement + sep)
            cont_prompts.extend([prompt] * args.k)
            meta.append((condition, burst_idx, burst, replacement, was_cut))

        flat = generate_batch_prompts(model, tokenizer, cont_prompts,
                                      args.continuation_tokens)

        for j, (condition, burst_idx, burst, replacement, was_cut) in enumerate(meta):
            continuations = flat[j * args.k:(j + 1) * args.k]

            results.append({
                "scenario_id": sid, "chunk_idx": i, "fraction": t.get("fraction"),
                "blackmail_rate_at_i": t.get("blackmail_rate"),
                "condition": condition, "layer": args.layer,
                "coefficient": args.coefficient, "steer_scope": args.steer_scope,
                "prefix_hash": prefix_hash, "burst_idx": burst_idx,
                "original_sentence": original, "replacement_sentence": replacement,
                "burst_was_cut": was_cut, "raw_burst": burst,
                "continuations": continuations,
            })

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    cut = sum(1 for r in results if r["burst_was_cut"])
    n_targets = len({(r["scenario_id"], r["chunk_idx"]) for r in results})
    print(f"\nwrote {args.out} ({len(results)} rows = "
          f"{n_targets} targets x {len(args.conditions)} conditions x {args.n_bursts} bursts)")
    print(f"boundary stopping worked in {cut}/{len(results)} bursts")
    print(f"steer_scope={args.steer_scope}"
          + ("" if args.steer_scope == "generated" else
             "  <-- steering the WHOLE prompt, not just the generated sentence"))
    if args.n_bursts == 1:
        print("NOTE: n_bursts=1 -- each S estimate rests on a single steered sentence, while R "
              "averages over ~100. Raise --n_bursts before comparing S to R.")
    print("Read the S replacement sentences: they must be fluent, reconsidering, and about the "
          "DECISION -- not meta-commentary about the prompt. Report the decision-relevant rate.")


if __name__ == "__main__":
    main()
