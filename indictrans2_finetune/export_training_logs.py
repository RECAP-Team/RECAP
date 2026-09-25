"""
Export a clean CSV of training/eval history for every IndicTrans2 LoRA job,
parsed from each job's trainer_state.json.

trainer_state.json accumulates the FULL log_history from step 0 onward even
though save_total_limit=1 prunes older checkpoint-N/ dirs as training goes --
the single surviving checkpoint always has the complete history, so this
works whether a job is still training or already finished.

Note: eval events happen every --eval_steps (default 1000), which does not
line up with epoch boundaries -- rows land at whatever fractional "epoch"
that step corresponds to (0.66, 1.3, 2.0, ...), not once per whole epoch.

Run (while jobs are still training, or after they finish):
    python export_training_logs.py --server pragya
    python export_training_logs.py --server server2 --out logs.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DIRECTIONS = ["hi2tgt", "tgt2hi"]


def find_trainer_state(job_dir: Path):
    checkpoints = sorted(
        job_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        return None
    return checkpoints[-1] / "trainer_state.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--server", default="pragya")
    ap.add_argument("--out", default=str(HERE / "training_logs_summary.csv"))
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    assert args.server in cfg["servers"], \
        f"unknown --server {args.server!r}, choices: {list(cfg['servers'])}"
    output_root = cfg["servers"][args.server]["output_root"]

    rows = []
    for lang in cfg["languages"]:
        for direction in DIRECTIONS:
            job_dir = Path(output_root) / lang / f"indictrans2-{lang.lower()}-{direction}-lora"
            ts_path = find_trainer_state(job_dir)
            if ts_path is None or not ts_path.exists():
                print(f"[skip] {lang}/{direction}: no checkpoint yet at {job_dir}")
                continue
            state = json.load(open(ts_path))
            for entry in state.get("log_history", []):
                rows.append({
                    "language": lang,
                    "direction": direction,
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "train_loss": entry.get("loss"),
                    "eval_loss": entry.get("eval_loss"),
                    "eval_BLEU": entry.get("eval_BLEU"),
                    "eval_chrF": entry.get("eval_chrF"),
                    "learning_rate": entry.get("learning_rate"),
                })
            print(f"[read] {lang}/{direction}: {len(state.get('log_history', []))} log entries "
                  f"(from {ts_path})")

    if not rows:
        print("No logs found for any job yet.")
        return

    df = pd.DataFrame(rows).sort_values(["language", "direction", "step"])
    df.to_csv(args.out, index=False)
    print(f"\n[done] wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
