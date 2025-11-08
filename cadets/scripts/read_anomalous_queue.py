#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ..config import ARTIFACT_DIR


def build_history_frame(history_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(history_dir.glob("*_history_list")):
        history_list = torch.load(path)
        for queue_id, queue in enumerate(history_list):
            for tw in queue:
                rows.append(
                    {
                        "file": path.name,
                        "queue": queue_id,
                        "index": tw.get("index"),
                        "name": tw.get("name"),
                        "loss": tw.get("loss"),
                        "nodeset": ";".join(map(str, sorted(tw.get("nodeset", [])))),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise *_history_list files into CSV/TSV.")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path(ARTIFACT_DIR),
        help="Directory containing *_history_list tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_dir = args.history_dir
    history_dir.mkdir(parents=True, exist_ok=True)

    df = build_history_frame(history_dir)
    if df.empty:
        print(f"No history_list files found under {history_dir}")
        return

    csv_path = history_dir / "all_history_list.csv"
    tsv_path = history_dir / "all_history_list.tsv"

    df.to_csv(csv_path, index=False)
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Wrote combined CSV to {csv_path}")
    print(f"Wrote combined TSV to {tsv_path}")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\n=== Preview of loaded history lists ===\n")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
