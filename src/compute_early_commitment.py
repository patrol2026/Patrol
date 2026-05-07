"""
compute_early_commitment.py
───────────────────────────
Empirical validation of PATROL's early-commitment hypothesis.

For every nodefense response we already have on disk, we slice it at increasing
token prefixes and call LlamaGuard-2-8B on each prefix. The output is a JSON
file with one (model, attack) → list-of-prompt-series structure that
plot_early_commitment.py turns into the two-panel figure.

Usage
─────
    python compute_early_commitment.py
        [--max_prompts 50]
        [--prefix_tokens 20 40 60 80 100 120 150 200 300]
        [--models llama2 llama3 vicuna]
        [--out figures/early_commitment.json]

Resume support: if --out already exists, cells (model, attacker) already
present in it are skipped. Delete the file or pass --overwrite to start fresh.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import transformers
from tqdm import tqdm

REPO_ROOT      = Path(__file__).parent.resolve()
DEFAULT_OUT    = REPO_ROOT / "figures" / "early_commitment.json"
GUARD_NAME     = "meta-llama/Meta-Llama-Guard-2-8B"
DEFAULT_PREFIX = [20, 40, 60, 80, 100, 120, 150, 200, 300]   # last entry "full" is appended

# Map result-folder name → model_name string used in filenames
SOURCE_DIRS = {
    "llama2": REPO_ROOT / "results_llama2",
    "llama3": REPO_ROOT / "results_llama",
    "vicuna": REPO_ROOT / "results_vicuna",
}

ATTACKS = ["AdvBench", "GCG", "AutoDAN", "PAIR", "DeepInception", "RandomSearch",
           "alpaca_eval"]   # alpaca_eval is the BENIGN control for FPR


# ─────────────────────────────────────────────────────────────────────────────
# LlamaGuard-2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_guard():
    print(f"Loading {GUARD_NAME} …", flush=True)
    tok   = transformers.AutoTokenizer.from_pretrained(GUARD_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        GUARD_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    return model, tok


@torch.no_grad()
def is_unsafe(goal: str, response: str, model, tok) -> bool:
    chat = [
        {"role": "user",      "content": goal},
        {"role": "assistant", "content": response},
    ]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    verdict = tok.decode(new_ids, skip_special_tokens=True).strip().lower()
    return verdict.startswith("unsafe")


def truncate_to_tokens(text: str, tok, n: int | None) -> str:
    """Use the LlamaGuard tokenizer (any modern BPE will do) to slice at token n."""
    if n is None:
        return text
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_SUFFIXES   = ("_judged.json", "_safe_eval.json", "_winrate.json",
                    "_successful_attacks.json")
_SKIP_SUBSTRINGS = ("_gpt41_judged",)   # alpaca_eval kept; benign control


def discover(model_name: str) -> dict[str, Path]:
    """Return {attacker: path} for nodefense files of one model."""
    src = SOURCE_DIRS[model_name]
    out = {}
    if not src.exists(): return out
    for f in sorted(src.iterdir()):
        if not f.name.startswith("nodefense_"): continue
        if any(f.name.endswith(s) for s in _SKIP_SUFFIXES):    continue
        # NOTE: do NOT skip alpaca_eval here; we need it as the benign control
        m = re.match(rf"nodefense_{model_name}_([A-Za-z_]+)_n", f.name)
        if not m: continue
        attacker = m.group(1)
        if attacker not in ATTACKS: continue
        out.setdefault(attacker, f)   # keep the first (oldest) match per attack
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_prompts",   type=int, default=50)
    p.add_argument("--prefix_tokens", type=int, nargs="+", default=DEFAULT_PREFIX)
    p.add_argument("--models",        nargs="+", default=list(SOURCE_DIRS),
                   choices=list(SOURCE_DIRS))
    p.add_argument("--out",           type=Path, default=DEFAULT_OUT)
    p.add_argument("--overwrite",     action="store_true",
                   help="Start fresh; do not resume from existing --out file.")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    if args.out.exists() and not args.overwrite:
        existing = json.loads(args.out.read_text())
        results  = existing["data"]
        prefix_tokens = existing["prefix_tokens"]
        if prefix_tokens != args.prefix_tokens + [None]:
            print("[warn] existing prefix_tokens differ; using existing layout.",
                  file=sys.stderr)
            args.prefix_tokens = [t for t in prefix_tokens if t is not None]
        print(f"Resuming from {args.out}: {len(results)} cells already done.")
    else:
        results = {}
        prefix_tokens = args.prefix_tokens + [None]   # None = full response

    # Plan
    plan = []
    for m in args.models:
        for a, path in discover(m).items():
            key = f"{m}|{a}"
            if key in results and len(results[key]) >= args.max_prompts:
                continue
            plan.append((m, a, path))

    if not plan:
        print("Nothing to do; output already complete.")
        return

    print("\nWill compute:")
    for m, a, path in plan:
        print(f"  {m}/{a:<14} ← {path.name}")
    print(f"\nPrefix tokens: {prefix_tokens}\n")

    model, tok = load_guard()

    for m, a, path in plan:
        key = f"{m}|{a}"
        items = json.loads(path.read_text()).get("data", [])
        items = items[:args.max_prompts]
        if not items: continue

        per_prompt = results.get(key, [])
        already_done = {p["goal"] for p in per_prompt}

        for it in tqdm(items, desc=f"{m}/{a}", leave=True):
            goal     = (it.get("goal") or "").strip()
            response = (it.get("output") or "").strip()
            if not goal or not response: continue
            if goal in already_done:
                continue
            series = []
            for n in prefix_tokens:
                pref = truncate_to_tokens(response, tok, n)
                if not pref:
                    series.append(False)
                    continue
                series.append(bool(is_unsafe(goal, pref, model, tok)))
            per_prompt.append({"goal": goal, "series": series})

            # Periodic checkpoint (every 5 prompts)
            if len(per_prompt) % 5 == 0:
                results[key] = per_prompt
                args.out.write_text(json.dumps(
                    {"prefix_tokens": prefix_tokens, "data": results}))

        results[key] = per_prompt
        args.out.write_text(json.dumps(
            {"prefix_tokens": prefix_tokens, "data": results}))

    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
