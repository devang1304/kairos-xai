"""Utility script to export node/event records to NDJSON."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from db_utils import run_sql


def fetch_node_metadata(node_ids: Iterable[int]) -> List[dict]:
    sql = """
        SELECT index_id, node_type, msg
        FROM node2id
        WHERE index_id = ANY(%s)
        ORDER BY index_id;
    """
    return run_sql(sql, (list(node_ids),)) or []


def fetch_events(node_ids: Iterable[int], limit: int) -> List[dict]:
    sql = """
        SELECT src, dst, relation_type, timestamp_rec, msg
        FROM event_table
        WHERE src = ANY(%s) OR dst = ANY(%s)
        ORDER BY timestamp_rec
        LIMIT %s;
    """
    ids = list(node_ids)
    return run_sql(sql, (ids, ids, limit)) or []


def export_to_ndjson(records: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_export(node_ids: List[int], limit: int, output: Path) -> None:
    if not node_ids:
        raise ValueError("At least one node id is required")

    nodes = fetch_node_metadata(node_ids)
    events = fetch_events(node_ids, limit)

    ndjson_records: List[dict] = []
    for row in nodes:
        ndjson_records.append({"type": "node", **row})
    for row in events:
        ndjson_records.append({"type": "event", **row})

    export_to_ndjson(ndjson_records, output)
    print(f"Wrote {len(ndjson_records)} records to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export node/event records to NDJSON.")
    parser.add_argument("--nodes", type=int, nargs="+", required=True, help="Node IDs to query.")
    parser.add_argument("--limit", type=int, default=100, help="Max number of event records to fetch.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifact/db_exports/node_event_records.ndjson"),
        help="Destination NDJSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_export(node_ids=args.nodes, limit=args.limit, output=args.output)
