"""
Phase 2 data tool: pull the QwQ-32B blackmail rollouts into the directory tree the
thought-branches loaders expect, and select target sentences.

The release (uzaymacar/blackmail-rollouts) is parquet rows of (path, content). Confirmed
layout for QwQ:

    qwq-32b/temperature_0.7_top_p_0.95/yes_base_solution/scenario_<id>/
        scenario.json            system_prompt, user_prompt, email_content
        base_solution.json       "solution" = base CoT trace
        chunks.json              "chunks", "separators"  (byte-faithful split)
        chunks_labeled.json      per-chunk: chunk, chunk_idx, blackmail_rate, function_tags, ...
        chunk_<i>/solutions.json  100 rollouts, each: chunk_resampled, rollout, full_cot,
                                  contains_blackmail (bool), prefix_without_chunk

Condition mapping (verified against bytes):
    R   = chunk_<i>/solutions.json          -- resamples of sentence i
    I0  = chunk_<i+1>/solutions.json        -- keep sentence i, continue; rate = P(bm|prefix[:i+1])
    P(blackmail|prefix at i) = chunks_labeled.json[i]["blackmail_rate"]   (precomputed!)

Because blackmail_rate is precomputed per chunk, TARGET SELECTION needs only the four small
files per scenario. The big solutions.json files are pulled lazily, only for chosen targets
(and i+1 for I0).

Runs with duckdb + httpfs, no GPU, no model. Subcommands:

    python fetch_blackmail_rollouts.py index     --out data/blackmail
    python fetch_blackmail_rollouts.py pull-small --out data/blackmail [--limit N]
    python fetch_blackmail_rollouts.py gate       --out data/blackmail --scenario 0
    python fetch_blackmail_rollouts.py targets    --out data/blackmail [--band 0.25 0.75] [...]
    python fetch_blackmail_rollouts.py pull-chunks --out data/blackmail   # from targets.json
"""

import argparse
import json
import os
import re
import sys

import duckdb

import _paths  # noqa: F401  -- resolves DATA_DIR

DATASET = "uzaymacar/blackmail-rollouts"
N_SHARDS = 92
SHARD_URL = ("https://huggingface.co/datasets/" + DATASET +
             "/resolve/main/data/default-{:05d}-of-{:05d}.parquet")
QWQ_PREFIX = "qwq-32b/temperature_0.7_top_p_0.95/yes_base_solution"
SMALL_FILES = ["scenario.json", "base_solution.json", "chunks.json", "chunks_labeled.json"]

# A sentence opening with one of these is already a reconsideration; steering reconsideration
# onto it has nothing to add. See is_reconsideration() in cmd_targets.
RECONSIDER_OPENER = re.compile(
    r"^\s*(but|however|alternatively|wait|actually|instead|maybe|perhaps|"
    r"on the other hand|then again|hmm|although|though|conversely|"
    r"that said|even so|still,)\b", re.I)


def con():
    c = duckdb.connect()
    c.execute("INSTALL httpfs; LOAD httpfs;")
    return c


def all_shards():
    return [SHARD_URL.format(i, N_SHARDS) for i in range(N_SHARDS)]


def write_file(out_dir, rel_path, content):
    dest = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(content)


# --- index: which shards hold which QwQ scenarios ----------------------------
def cmd_index(args):
    """Scan every shard's path column (cheap projection) and map scenario -> shard."""
    c = con()
    manifest = {}          # scenario_id -> shard index holding its files
    for i, url in enumerate(all_shards()):
        try:
            rows = c.execute(
                f"SELECT DISTINCT regexp_extract(path,'scenario_(\\d+)',1) AS sid "
                f"FROM read_parquet('{url}') WHERE path LIKE '{QWQ_PREFIX}/%'"
            ).fetchall()
        except Exception as exc:
            print(f"shard {i}: ERR {exc}")
            continue
        sids = [r[0] for r in rows if r[0]]
        if sids:
            for sid in sids:
                manifest.setdefault(sid, []).append(i)
            print(f"shard {i:2d}: {len(sids)} qwq scenarios")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "shard_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} QwQ scenarios across shards; wrote {path}")


def load_manifest(out_dir):
    path = os.path.join(out_dir, "shard_manifest.json")
    if not os.path.exists(path):
        sys.exit(f"No manifest at {path}. Run `index` first.")
    with open(path) as f:
        return json.load(f)


# --- pull-small: the four small files per scenario ---------------------------
def cmd_pull_small(args):
    manifest = load_manifest(args.out)
    c = con()
    sids = sorted(manifest, key=lambda s: int(s))
    if args.limit:
        sids = sids[:args.limit]
    for sid in sids:
        # A scenario's files can span several shards (its ~120 chunk files plus the small
        # files do not fit in one row-batch), so search every shard the index recorded.
        like = f"{QWQ_PREFIX}/scenario_{sid}/%"
        got = set()
        for shard_idx in manifest[sid]:
            shard = SHARD_URL.format(shard_idx, N_SHARDS)
            for fname in SMALL_FILES:
                if fname in got:
                    continue
                rows = c.execute(
                    f"SELECT path, content FROM read_parquet('{shard}') "
                    f"WHERE path LIKE '{like}' AND path LIKE '%/{fname}'"
                ).fetchall()
                for path, content in rows:
                    if content is not None:
                        write_file(args.out, path, content)
                        got.add(fname)
        missing = set(SMALL_FILES) - got
        print(f"scenario {sid}: {len(got)} small files" +
              (f"  MISSING {sorted(missing)}" if missing else ""))
    print(f"\npulled small files for {len(sids)} scenarios into {args.out}")


# --- gate: assert the schema on one scenario ---------------------------------
def cmd_gate(args):
    base = os.path.join(args.out, QWQ_PREFIX, f"scenario_{args.scenario}")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    cl_path = os.path.join(base, "chunks_labeled.json")
    check(os.path.exists(cl_path), f"chunks_labeled.json present ({cl_path})")
    if os.path.exists(cl_path):
        cl = json.load(open(cl_path))
        check(isinstance(cl, list) and len(cl) > 5, f"chunks_labeled is a list ({len(cl)} chunks)")
        c0 = cl[0]
        check("blackmail_rate" in c0, "per-chunk blackmail_rate present")
        check("function_tags" in c0, "per-chunk function_tags present")
        check("chunk" in c0 and "chunk_idx" in c0, "chunk text + idx present")

    for fname in ("scenario.json", "base_solution.json", "chunks.json"):
        p = os.path.join(base, fname)
        check(os.path.exists(p), f"{fname} present")
    bs = os.path.join(base, "base_solution.json")
    if os.path.exists(bs):
        check("solution" in json.load(open(bs)), "base_solution has 'solution'")

    # If any chunk solutions.json has been pulled, verify per-rollout schema.
    chunk_dirs = [d for d in (os.listdir(base) if os.path.exists(base) else []) if d.startswith("chunk_")]
    if chunk_dirs:
        sp = os.path.join(base, sorted(chunk_dirs)[0], "solutions.json")
        if os.path.exists(sp):
            sols = json.load(open(sp))
            check(isinstance(sols, list) and len(sols) > 10, f"solutions.json is a list ({len(sols)})")
            e = sols[0]
            for k in ("chunk_resampled", "rollout", "contains_blackmail", "prefix_without_chunk"):
                check(k in e, f"rollout has '{k}'")
            check(isinstance(e.get("contains_blackmail"), bool), "contains_blackmail is bool")
    else:
        print("  (no chunk_* pulled yet; run pull-chunks to verify per-rollout schema)")

    print("\n" + ("GATE PASSED" if ok else "GATE FAILED"))
    sys.exit(0 if ok else 1)


# --- targets: pick sentences from the small files ----------------------------
def cmd_targets(args):
    manifest = load_manifest(args.out)
    lo, hi = args.band
    targets = []
    for sid in sorted(manifest, key=lambda s: int(s)):
        cl_path = os.path.join(args.out, QWQ_PREFIX, f"scenario_{sid}", "chunks_labeled.json")
        if not os.path.exists(cl_path):
            continue
        cl = json.load(open(cl_path))
        n = len(cl)
        # index by chunk_idx for tag lookups on neighbours
        by_idx = {c.get("chunk_idx", j): c for j, c in enumerate(cl)}

        def is_reconsideration(chunk):
            """Is this sentence already a reconsideration?

            Checking function_tags alone is NOT enough: the released tag vocabulary has no
            backtrack/reconsider category at all (it is plan_generation,
            situation_assessment, leverage_identification, email_analysis,
            action_execution, ...), so a tags-only test never fires. Measured consequence:
            21 of 57 targets were sentences already opening with "But" / "Alternatively" /
            "Wait", and steering reconsideration onto those regenerates the original --
            observed at scenario 0 chunks 50 and 56.

            So also test the text for a leading reconsideration marker.
            """
            tags = chunk.get("function_tags") or []
            if any(("backtrack" in str(t).lower() or "reconsider" in str(t).lower()) for t in tags):
                return True
            return bool(RECONSIDER_OPENER.match(chunk.get("chunk") or ""))

        live = []
        for j, chunk in enumerate(cl):
            idx = chunk.get("chunk_idx", j)
            rate = chunk.get("blackmail_rate")
            if rate is None or not (lo <= rate <= hi):
                continue
            # not itself or its neighbours a backtrack
            if any(is_reconsideration(by_idx.get(idx + d, {})) for d in (-1, 0, 1)):
                continue
            live.append({
                "idx": idx, "rate": rate, "frac": idx / max(1, n - 1),
                "ci": chunk.get("counterfactual_importance_blackmail_rate"),
            })

        if not live:
            continue

        def emit(cand, selection):
            # function_tags and the original text travel with the target so phase2_generate
            # can filter by tag (--tag plan_generation) without re-reading chunks_labeled.
            chunk_meta = by_idx.get(cand["idx"], {})
            targets.append({
                "scenario_id": sid, "chunk_idx": cand["idx"],
                "blackmail_rate": cand["rate"], "fraction": round(cand["frac"], 3),
                "counterfactual_importance": cand["ci"], "selection": selection,
                "function_tags": chunk_meta.get("function_tags") or [],
                "original_sentence": chunk_meta.get("chunk") or "",
            })

        chosen_idxs = []
        if args.strategy in ("positional", "both"):
            # the live positions closest to the requested fractions
            for want in args.positions:
                cand = min(live, key=lambda c: abs(c["frac"] - want))
                if cand["idx"] not in chosen_idxs:
                    chosen_idxs.append(cand["idx"])
                    emit(cand, "positional")

        if args.strategy in ("importance", "both"):
            # Highest counterfactual importance WITHIN the same band, and not adjacent to a
            # positional pick. Restricting to the band matters: the globally most important
            # chunk is often at a near-decided prefix, so an unrestricted pick would differ
            # from the positional arm on "how live the decision is" as well as on importance,
            # and the contrast would be uninterpretable.
            cands = [c for c in live
                     if c["ci"] is not None
                     and all(abs(c["idx"] - p) > 1 for p in chosen_idxs)]
            if cands:
                emit(max(cands, key=lambda c: c["ci"]), "importance")

    path = os.path.join(args.out, "targets.json")
    with open(path, "w") as f:
        json.dump(targets, f, indent=2)
    n_traces = len({t["scenario_id"] for t in targets})
    by_sel = {}
    for t in targets:
        by_sel.setdefault(t["selection"], []).append(t)
    print(f"selected {len(targets)} (trace, position) pairs across {n_traces} traces "
          f"[band {lo}-{hi}, positions {args.positions}, strategy {args.strategy}]")
    for sel, rows in sorted(by_sel.items()):
        cis = [r["counterfactual_importance"] for r in rows
               if r["counterfactual_importance"] is not None]
        med = sorted(cis)[len(cis) // 2] if cis else float("nan")
        print(f"  {sel:12s} n={len(rows):3d}  median counterfactual importance {med:.3f}")
    print(f"wrote {path}")
    if len(targets) < 50:
        print("WARNING: fewer than ~50 pairs. Relax --band before relaxing anything else.")


# --- pull-chunks: the big solutions.json for targets (and i+1 for I0) --------
def cmd_pull_chunks(args):
    manifest = load_manifest(args.out)
    path = os.path.join(args.out, "targets.json")
    if not os.path.exists(path):
        sys.exit("No targets.json. Run `targets` first.")
    targets = json.load(open(path))
    c = con()
    # collect (scenario, chunk_idx) needing a pull: target i AND i+1 (for I0)
    need = {}
    for t in targets:
        sid = t["scenario_id"]
        need.setdefault(sid, set()).update({t["chunk_idx"], t["chunk_idx"] + 1})
    for sid, idxs in need.items():
        shards = manifest[sid]
        for idx in sorted(idxs):
            rel = f"{QWQ_PREFIX}/scenario_{sid}/chunk_{idx}/solutions.json"
            dest = os.path.join(args.out, rel)
            if os.path.exists(dest) and not args.force:
                continue
            found = False
            for shard_idx in shards:      # a scenario spans shards; try each
                shard = SHARD_URL.format(shard_idx, N_SHARDS)
                row = c.execute(
                    f"SELECT content FROM read_parquet('{shard}') WHERE path = '{rel}'"
                ).fetchone()
                if row and row[0]:
                    write_file(args.out, rel, row[0])
                    print(f"scenario {sid} chunk {idx}: pulled (shard {shard_idx})")
                    found = True
                    break
            if not found:
                print(f"scenario {sid} chunk {idx}: MISSING across shards {shards}")
    print("done")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index"); pi.set_defaults(fn=cmd_index)
    pi.add_argument("--out", default=str(_paths.DATA_DIR))

    ps = sub.add_parser("pull-small"); ps.set_defaults(fn=cmd_pull_small)
    ps.add_argument("--out", default=str(_paths.DATA_DIR))
    ps.add_argument("--limit", type=int, default=None)

    pg = sub.add_parser("gate"); pg.set_defaults(fn=cmd_gate)
    pg.add_argument("--out", default=str(_paths.DATA_DIR))
    pg.add_argument("--scenario", type=str, required=True)

    pt = sub.add_parser("targets"); pt.set_defaults(fn=cmd_targets)
    pt.add_argument("--out", default=str(_paths.DATA_DIR))
    pt.add_argument("--band", type=float, nargs=2, default=[0.25, 0.75])
    pt.add_argument("--positions", type=float, nargs="+", default=[0.3, 0.6])
    pt.add_argument("--strategy", choices=["positional", "importance", "both"], default="both",
                    help="positional: sentences at the requested trace fractions. importance: "
                         "the highest counterfactual-importance sentence within the same band. "
                         "both (default) emits each arm, tagged in the 'selection' field, for "
                         "the strategic-vs-positional targeting experiment.")

    pc = sub.add_parser("pull-chunks"); pc.set_defaults(fn=cmd_pull_chunks)
    pc.add_argument("--out", default=str(_paths.DATA_DIR))
    pc.add_argument("--force", action="store_true", default=False)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
