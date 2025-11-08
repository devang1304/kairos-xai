"""Export nodeid→message mapping to JSON using DB credentials from config.py."""
from __future__ import annotations

import json
from pathlib import Path

import psycopg2

from ..config import DATABASE, HOST, NODE_MAPPING_JSON, PASSWORD, PORT, USER

OUTPUT_PATH = Path(NODE_MAPPING_JSON)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with psycopg2.connect(
        dbname=DATABASE,
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
    ) as conn, conn.cursor() as cur:
        cur.execute("SELECT index_id, node_type, msg FROM node2id ORDER BY index_id")
        rows = cur.fetchall()

    mapping = {int(index_id): f"{node_type}: {msg}" for index_id, node_type, msg in rows}
    OUTPUT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(mapping)} entries to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
