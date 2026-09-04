"""
Inspect a trained mean_vectors_*.pt before spending GPU time on a steering sweep.

Answers three questions, in order of how badly a "no" would hurt:

  1. Is the vector estimable at all?  -- counts per label, finite values
  2. Is there a direction to speak of? -- ||feature|| / ||overall|| across layers. A flat
     profile means the labelled spans are not distinguishable from the mean thinking state
     and no layer will steer.
  3. Is it specific to backtracking?   -- cosine to the other label features. If backtracking
     is collinear with, say, uncertainty-estimation everywhere, you are steering "reasoning
     intensity", not backtracking.

Default mode needs only the .pt file -- runs on a laptop, no GPU, no weights.

  python validate_vectors.py --model Qwen/QwQ-32B

With --unembed it also projects the feature through the model's unembedding to show which
tokens each layer's direction promotes. This is the single most convincing check that the
direction means what its label says -- you want to see "Wait", "Alternatively", "However".
It reads lm_head straight out of the local safetensors shards, so it needs the weights
present but never builds the model:

  python validate_vectors.py --model Qwen/QwQ-32B --unembed
"""

import argparse
import os

import _paths  # noqa: F401  -- resolves VECTORS_DIR, .env

import torch

parser = argparse.ArgumentParser(description="Validate trained steering vectors")
parser.add_argument("--model", type=str, default="Qwen/QwQ-32B")
parser.add_argument("--vectors_path", type=str, default=None)
parser.add_argument("--label", type=str, default="backtracking",
                    help="Label to focus the report on")
parser.add_argument("--top_layers", type=int, default=6,
                    help="How many candidate layers to shortlist")
parser.add_argument("--unembed", action="store_true", default=False,
                    help="Also project features through the unembedding (needs local weights)")
parser.add_argument("--top_k", type=int, default=12,
                    help="Tokens to show per layer with --unembed")
parser.add_argument("--vocab_min_count", type=int, default=3,
                    help="With --unembed, only consider tokens appearing at least this many "
                         "times in the model's own generated responses. Rare tokens have "
                         "idiosyncratic high-variance unembedding rows and dominate the "
                         "top-k for almost any direction. 0 disables the filter.")
parser.add_argument("--orthogonalize", type=str, default=None,
                    help="Project this label's direction OUT of the target before reporting. "
                         "Use when the target is collinear with a confound, e.g. "
                         "--orthogonalize uncertainty-estimation.")
parser.add_argument("--edge_layers", type=int, default=1,
                    help="Layers to exclude at each end when shortlisting candidates. The "
                         "final layer feeds straight into the norm+unembedding and its "
                         "residual statistics differ, so it shows a large spurious spike.")
args, _ = parser.parse_known_args()

LABELS = ["initializing", "deduction", "adding-knowledge",
          "example-testing", "uncertainty-estimation", "backtracking"]

path = args.vectors_path or \
    str(_paths.VECTORS_DIR / f"mean_vectors_{args.model.split('/')[-1].lower()}.pt")
raw = torch.load(path, map_location="cpu")

print(f"{path}\n")

# --- 1. estimable? -----------------------------------------------------------
print("=== counts ===")
for key in ["overall"] + LABELS:
    if key not in raw:
        print(f"  {key:26s} MISSING")
        continue
    mean, count = raw[key]["mean"].float(), raw[key]["count"]
    flag = ""
    if not torch.isfinite(mean).all():
        flag = "  <-- NON-FINITE VALUES"
    elif key == args.label and count < 20:
        flag = "  <-- too few to trust"
    print(f"  {key:26s} count={count:5d} shape={tuple(mean.shape)}{flag}")

overall = raw["overall"]["mean"].float()
n_layers = overall.shape[0]
features = {l: raw[l]["mean"].float() - overall for l in LABELS if l in raw}

# Optionally remove a confound direction layer by layer. The label features all share a
# large common component (every one is `label - overall`), so a target can look collinear
# with a confound while still carrying a real, smaller, specific part. Projecting the
# confound out isolates that part -- and if almost nothing survives, that is itself the
# answer.
if args.orthogonalize:
    confound = features[args.orthogonalize]
    target = features[args.label]
    residual = torch.empty_like(target)
    kept = []
    for layer in range(n_layers):
        c = confound[layer] / confound[layer].norm()
        residual[layer] = target[layer] - (target[layer] @ c) * c
        kept.append((residual[layer].norm() / target[layer].norm()).item())
    features[args.label] = residual
    print(f"\n=== projected out '{args.orthogonalize}' from '{args.label}' ===")
    print(f"  fraction of the original norm surviving: "
          f"min={min(kept):.2f} median={sorted(kept)[len(kept)//2]:.2f} max={max(kept):.2f}")
    if max(kept) < 0.35:
        print("  WARNING: little survives at any layer -- the two labels are not separable "
              "in this estimate, and steering one steers the other.")

# --- 2. is there a direction? ------------------------------------------------
# Ratio of the label-specific deviation to the typical activation magnitude at that layer.
# The paper's steering normalises the feature to ||overall|| before applying it, so this
# ratio is effectively the signal-to-background of the raw estimate.
ratio = torch.stack([
    features[args.label][layer].norm() / overall[layer].norm() for layer in range(n_layers)
])

print(f"\n=== ||{args.label} - overall|| / ||overall|| across {n_layers} layers ===")
peak = ratio.max().item()
for layer in range(n_layers):
    r = ratio[layer].item()
    bar = "#" * int(round(40 * r / peak)) if peak > 0 else ""
    print(f"  L{layer:02d} {r:6.3f} {bar}")

flatness = ratio.std().item() / max(ratio.mean().item(), 1e-9)
print(f"\n  peak {peak:.3f} at layer {int(ratio.argmax())} | "
      f"relative spread {flatness:.2f}")
if flatness < 0.15:
    print("  WARNING: profile is nearly flat -- the labelled spans barely differ from the "
          "mean thinking state. A steering sweep is unlikely to find anything.")

# --- 3. is it specific? ------------------------------------------------------
others = [l for l in features if l != args.label]
print(f"\n=== cosine({args.label}, other) per layer -- lower is more specific ===")
print(f"  {'layer':>5s} " + " ".join(f"{l[:11]:>11s}" for l in others) + "   max")
rows = {}
for layer in range(n_layers):
    cos = {l: torch.cosine_similarity(features[args.label][layer],
                                      features[l][layer], dim=0).item() for l in others}
    rows[layer] = cos
for layer in range(0, n_layers, 4):
    cos = rows[layer]
    print(f"  L{layer:02d}   " + " ".join(f"{cos[l]:11.2f}" for l in others)
          + f"   {max(cos.values()):5.2f}")

worst = max(others, key=lambda l: sum(rows[layer][l] for layer in range(n_layers)))
avg = sum(rows[layer][worst] for layer in range(n_layers)) / n_layers
print(f"\n  most collinear label overall: {worst} (mean cosine {avg:.2f})")
if avg > 0.8:
    print(f"  WARNING: '{args.label}' and '{worst}' point in nearly the same direction at "
          f"every layer. Steering one plausibly steers the other, and they cannot be "
          f"claimed as distinct behaviours without more evidence. Re-run with "
          f"--orthogonalize {worst}, and check --unembed to see which behaviour's "
          f"vocabulary the direction actually promotes.")

# --- candidate layers --------------------------------------------------------
# Want a large deviation that is NOT shared with the other behaviours: score = ratio scaled
# down by how collinear the direction is with the most similar other label.
score = torch.tensor([
    ratio[layer].item() * (1.0 - max(max(rows[layer].values()), 0.0))
    for layer in range(n_layers)
])
# Exclude the edge layers before ranking. The last layer sits after all the computation and
# immediately before the final norm + unembedding, so its residual norm is on a different
# scale -- it reliably wins this ranking on an artifact, and steering there would push output
# tokens directly rather than change the reasoning.
for layer in list(range(args.edge_layers)) + list(range(n_layers - args.edge_layers, n_layers)):
    score[layer] = float("-inf")
best = torch.argsort(score, descending=True)[:args.top_layers].tolist()
print(f"\n=== candidate layers for the sweep (norm ratio x specificity) ===")
for layer in best:
    print(f"  L{layer:02d}  ratio={ratio[layer]:.3f}  "
          f"max_cos={max(rows[layer].values()):.2f}  score={score[layer]:.3f}")
print(f"\n  suggested sweep: layers {sorted(best[:4])}"
      f"   (excluded {args.edge_layers} layer(s) at each end as boundary artifacts)")

# --- 4. optional: what tokens does the direction promote? --------------------
if args.unembed:
    import glob, json
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(cache, "hub", f"models--{args.model.replace('/', '--')}",
                           "snapshots", "*", "*.safetensors")
    shards = glob.glob(pattern)
    if not shards:
        raise FileNotFoundError(f"No safetensors found under {pattern}. Set HF_HOME.")

    # Which matrix? QwQ-32B sets tie_word_embeddings=false, so lm_head.weight and
    # model.embed_tokens.weight are DIFFERENT matrices and only lm_head is the unembedding.
    # Taking whichever appears first in shard order silently grabs embed_tokens (it lives in
    # shard 1, lm_head in the last) and produces meaningless top-token lists. So search every
    # shard for lm_head first, and fall back to embed_tokens only when the config actually
    # says the weights are tied.
    config_path = os.path.join(os.path.dirname(shards[0]), "config.json")
    tied = False
    if os.path.exists(config_path):
        with open(config_path) as f:
            tied = bool(json.load(f).get("tie_word_embeddings", False))

    def find_tensor(name):
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                if name in f.keys():
                    return f.get_tensor(name).float(), os.path.basename(shard)
        return None, None

    lm_head, shard_name = find_tensor("lm_head.weight")
    if lm_head is None:
        if not tied:
            raise RuntimeError(
                "lm_head.weight not found and tie_word_embeddings is false. Refusing to "
                "substitute model.embed_tokens.weight -- it is the INPUT embedding and "
                "projecting through it gives meaningless results."
            )
        lm_head, shard_name = find_tensor("model.embed_tokens.weight")
    print(f"\n  unembedding: {'tied embed_tokens' if lm_head is None else 'lm_head.weight'} "
          f"from {shard_name}")

    # Logit lens: apply the final RMSNorm before unembedding. A mid-stack residual is not in
    # the final-layer basis, and skipping the norm makes the top tokens much noisier.
    final_norm, _ = find_tensor("model.norm.weight")
    if final_norm is None:
        print("  (model.norm.weight not found -- projecting without the final norm)")

    # Centering the unembedding removes the component shared by every token, which otherwise
    # lets rows with unusual norms dominate the ranking regardless of direction.
    lm_head = lm_head - lm_head.mean(dim=0, keepdim=True)

    # Frequency mask built from the model's own outputs. QwQ uses a small fraction of the
    # 152k vocabulary here, and the unused remainder is where the rare-token artifact lives.
    vocab_mask = None
    if args.vocab_min_count > 0:
        responses_path = os.path.join(
            os.path.dirname(path), f"responses_{args.model.split('/')[-1].lower()}.json")
        if os.path.exists(responses_path):
            from collections import Counter
            counts = Counter()
            with open(responses_path) as f:
                for record in json.load(f):
                    counts.update(record.get("response_token_ids") or [])
            vocab_mask = torch.zeros(lm_head.shape[0], dtype=torch.bool)
            for token_id, count in counts.items():
                if count >= args.vocab_min_count and token_id < lm_head.shape[0]:
                    vocab_mask[token_id] = True
            print(f"  vocabulary restricted to {int(vocab_mask.sum())} tokens seen "
                  f">= {args.vocab_min_count}x in {os.path.basename(responses_path)}")
        else:
            print(f"  (no responses file at {responses_path}; skipping frequency filter)")

    def to_logits(v):
        x = v.float()
        if final_norm is not None:
            eps = 1e-5   # Qwen2 rms_norm_eps
            x = x / torch.sqrt(x.pow(2).mean() + eps) * final_norm
        return lm_head @ (x / x.norm())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"\n=== tokens promoted by the {args.label} direction ===")
    for layer in sorted(best[:4]):
        logits = to_logits(features[args.label][layer])
        if vocab_mask is not None:
            logits = logits.masked_fill(~vocab_mask, float("-inf"))
        top = torch.topk(logits, args.top_k).indices.tolist()
        toks = [tokenizer.decode([t]).strip() or repr(tokenizer.decode([t])) for t in top]
        print(f"  L{layer:02d}: {', '.join(toks)}")
    print("\n  Reconsideration words here ('Wait', 'Alternatively', 'However', 'But') are "
          "strong evidence the direction is what its label claims. Punctuation and "
          "whitespace fragments are weak evidence either way.")
