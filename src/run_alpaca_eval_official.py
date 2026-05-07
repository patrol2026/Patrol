"""Convert our defender output JSONs to AlpacaEval-2.0 format and judge them
with the official length-controlled GPT-4-Turbo annotator.

Each input file is split into:
    <stem>_outputs.json    — model responses (instruction, output, generator)
    <stem>_reference.json  — text-davinci-003 reference (from `reference_output`)
The pair is then judged via alpaca_eval's `weighted_alpaca_eval_gpt4_turbo`
config and the resulting leaderboard CSV is saved next to each pair.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def convert_one(src: Path, out_dir: Path, generator_name: str | None = None
                ) -> tuple[Path, Path] | None:
    """Split one of our result JSONs into AE-format outputs and reference."""
    raw = json.loads(src.read_text(encoding="utf-8"))
    items = raw.get("data", []) if isinstance(raw, dict) else raw
    if not items:
        print(f"  [skip] {src.name}: no data")
        return None

    if generator_name is None:
        ev = raw.get("experiment_variables", {}) if isinstance(raw, dict) else {}
        generator_name = ev.get("defender", src.stem)

    model_outputs, ref_outputs = [], []
    for it in items:
        instr = it.get("instruction", "")
        mout  = it.get("output", "")
        rout  = it.get("reference_output", "")
        if not instr or not mout or not rout:
            continue
        model_outputs.append({
            "instruction": instr,
            "output":      mout,
            "generator":   generator_name,
            "dataset":     it.get("dataset", "alpaca_eval"),
        })
        ref_outputs.append({
            "instruction": instr,
            "output":      rout,
            "generator":   "text-davinci-003",
            "dataset":     it.get("dataset", "alpaca_eval"),
        })

    if not model_outputs:
        print(f"  [skip] {src.name}: no usable rows")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / f"{src.stem}_outputs.json"
    r_path = out_dir / f"{src.stem}_reference.json"
    m_path.write_text(json.dumps(model_outputs, indent=2))
    r_path.write_text(json.dumps(ref_outputs,   indent=2))
    return m_path, r_path


def evaluate_one(m_path: Path, r_path: Path, annotators_config: str,
                 leaderboard_dir: Path) -> int:
    """Run `alpaca_eval evaluate` on one (outputs, reference) pair."""
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "alpaca_eval", "evaluate",
        "--model_outputs",      str(m_path),
        "--reference_outputs",  str(r_path),
        "--annotators_config",  annotators_config,
        "--output_path",        str(leaderboard_dir / m_path.stem),
        "--is_overwrite_leaderboard",
    ]
    print("  $", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                    help="Glob patterns of source result JSONs.")
    ap.add_argument("--output_dir", type=Path,
                    default=Path("results_alpaca_eval_official"))
    ap.add_argument("--annotators_config", default="weighted_alpaca_eval_gpt4_turbo",
                    help="alpaca_eval annotators_config name (default: AE 2.0 length-controlled).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip pairs whose leaderboard CSV already exists.")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY before running.")

    # Resolve inputs
    paths = []
    for pat in args.input:
        m = sorted(glob.glob(pat))
        paths.extend(Path(p) for p in m)
    paths = [p for p in paths
             if p.exists()
             and not p.name.endswith("_winrate.json")
             and not p.name.endswith("_safe_eval.json")
             and not p.name.endswith("_judged.json")
             and not p.name.endswith("_outputs.json")
             and not p.name.endswith("_reference.json")]

    print(f"Found {len(paths)} input file(s).")

    convert_dir = args.output_dir / "ae_inputs"
    leaderboard_dir = args.output_dir / "leaderboards"

    for src in paths:
        print(f"\n→ {src.name}")
        out_csv = leaderboard_dir / src.stem / "leaderboard.csv"
        if args.skip_existing and out_csv.exists():
            print(f"  [skip] already judged → {out_csv}")
            continue
        pair = convert_one(src, convert_dir)
        if pair is None:
            continue
        rc = evaluate_one(*pair, args.annotators_config, leaderboard_dir)
        if rc != 0:
            print(f"  [error] alpaca_eval exited with code {rc}")


if __name__ == "__main__":
    main()
