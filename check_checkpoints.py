#!/usr/bin/env python3
"""
Integrity check for every finetuned checkpoint / LoRA adapter under this
repo (correct_checkpoints/<model>/<language>/..., and the original Marathi
checkpoints under <model>_finetune/Marathi/, NLLB-finetune/Marathi/, etc.)
-- without loading any model into memory.

For each checkpoint directory (one ending in -finetuned, -lora, or a
-smoketest variant of either):
  - resolves to the actual save to check: the top-level dir if it has a
    finished adapter_config.json/config.json, otherwise the latest
    checkpoint-N/ subdir (same fallback the inference scripts use when a
    PBS job got walltime-killed before the final save)
  - every *.json file in that dir must parse as valid JSON
  - every *.safetensors file's header must read cleanly (safetensors
    safe_open -- catches truncated/corrupted saves without loading tensors)
  - every *.bin weight file must be a valid zip (torch.save format) and
    pass zipfile's CRC check (testzip)
  - a weight file (model.safetensors / pytorch_model.bin / sharded
    *.safetensors, or adapter_model.safetensors / adapter_model.bin for
    LoRA) must actually be present

Run (no GPU needed, plain python3 + safetensors):
    python3 check_checkpoints.py [root]
(root defaults to this script's own directory, i.e. the repo root)

Exit code is 0 if everything is clean, 1 if any problem was found.
"""
import json
import sys
import zipfile
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
LEAF_SUFFIXES = ("-finetuned", "-lora", "-finetuned-smoketest", "-lora-smoketest")

FULL_SAVE_WEIGHTS = ["model.safetensors", "pytorch_model.bin"]
LORA_WEIGHTS = ["adapter_model.safetensors", "adapter_model.bin"]


def check_json(path, problems):
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        problems.append(f"BAD JSON {path}: {e}")


def check_safetensors(path, problems):
    if safe_open is None:
        problems.append(f"SKIPPED (pip install safetensors to check) {path}")
        return
    try:
        with safe_open(str(path), framework="pt") as f:
            if not list(f.keys()):
                problems.append(f"EMPTY safetensors {path}")
    except Exception as e:
        problems.append(f"CORRUPT safetensors {path}: {e}")


def check_bin_zip(path, problems):
    try:
        if not zipfile.is_zipfile(path):
            problems.append(f"NOT A VALID torch.save ZIP {path}")
            return
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                problems.append(f"CORRUPT ENTRY in {path}: {bad}")
    except Exception as e:
        problems.append(f"CORRUPT .bin {path}: {e}")


def resolve_leaf(leaf_dir):
    if (leaf_dir / "adapter_config.json").exists() or (leaf_dir / "config.json").exists():
        return leaf_dir
    ckpts = sorted(leaf_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    return ckpts[-1] if ckpts else None


def check_leaf(leaf_dir, problems, warnings):
    target = resolve_leaf(leaf_dir)
    if target is None:
        problems.append(f"NO FINISHED SAVE OR checkpoint-N/ found under {leaf_dir}")
        return
    if target != leaf_dir:
        warnings.append(f"no top-level save, using fallback {target.name}/")

    is_lora = (target / "adapter_config.json").exists()
    required = ["adapter_config.json"] if is_lora else ["config.json"]
    weight_candidates = LORA_WEIGHTS if is_lora else FULL_SAVE_WEIGHTS

    for req in required:
        p = target / req
        if not p.exists():
            problems.append(f"MISSING {p}")
        else:
            check_json(p, problems)

    has_weight = False
    for wname in weight_candidates:
        wp = target / wname
        if wp.exists():
            has_weight = True
            check_safetensors(wp, problems) if wp.suffix == ".safetensors" else check_bin_zip(wp, problems)

    shards = [s for s in target.glob("*.safetensors") if s.name not in weight_candidates]
    for s in shards:
        has_weight = True
        check_safetensors(s, problems)

    if not has_weight:
        problems.append(f"NO WEIGHT FILE in {target} (expected one of {weight_candidates} or shards)")

    for jf in target.glob("*.json"):
        if jf.name not in required:
            check_json(jf, problems)


def main():
    leaves = sorted(p for p in ROOT.rglob("*") if p.is_dir() and p.name.endswith(LEAF_SUFFIXES))

    print(f"[root] {ROOT}")
    print(f"[found] {len(leaves)} checkpoint directories\n")

    total_problems = 0
    for leaf in leaves:
        problems, warnings = [], []
        check_leaf(leaf, problems, warnings)
        rel = leaf.relative_to(ROOT)
        if problems:
            print(f"XX  {rel}")
            for w in warnings:
                print(f"     (note) {w}")
            for pr in problems:
                print(f"     !! {pr}")
            total_problems += len(problems)
        elif warnings:
            print(f"ok  {rel}  (fallback used)")
            for w in warnings:
                print(f"     (note) {w}")
        else:
            print(f"ok  {rel}")

    print(f"\n{'ALL CLEAR' if total_problems == 0 else f'{total_problems} PROBLEM(S) FOUND'} across {len(leaves)} checkpoints")
    sys.exit(1 if total_problems else 0)


if __name__ == "__main__":
    main()
