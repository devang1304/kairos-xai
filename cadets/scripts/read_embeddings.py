from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.temporal import TemporalData

from ..config import ARTIFACT_DIR

torch.serialization.add_safe_globals([TemporalData, GlobalStorage])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TemporalData embeddings and export CSVs.")
    parser.add_argument(
        "--embed-dir",
        type=Path,
        default=Path(ARTIFACT_DIR) / "graph_embeddings",
        help="Directory containing *.TemporalData.simple files.",
    )
    return parser.parse_args()


def process_file(path: Path) -> None:
    data = torch.load(path, weights_only=True)
    src, dst, t, msg = data.src, data.dst, data.t, data.msg
    print(f"\n=== {path.name} ===")
    print(f"Edges: {src.size(0)}  |  Features: {msg.size(1)}-dim")
    print(" src:", src[:5].tolist())
    print(" dst:", dst[:5].tolist())
    print("  t :", t[:5].tolist())
    print("msg shape:", msg[:5].shape)

    unique_nodes = set(src.tolist()) | set(dst.tolist())
    print(f"Unique nodes (src + dst): {len(unique_nodes)}")

    df = pd.DataFrame(
        {
            "src": src.tolist(),
            "dst": dst.tolist(),
            "t": t.tolist(),
            **{f"feat_{i}": msg[:, i].tolist() for i in range(msg.size(1))},
        }
    )
    print(df.head())

    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV to {csv_path}")


def main() -> None:
    args = parse_args()
    embed_dir = args.embed_dir
    files = sorted(embed_dir.glob("*.TemporalData.simple"))
    if not files:
        print(f"No embeddings found in {embed_dir}")
        return
    for path in files:
        process_file(path)


if __name__ == "__main__":
    main()
