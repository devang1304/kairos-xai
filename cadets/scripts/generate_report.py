"""Generate Markdown/HTML reports for a given explanation JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config import NODE_MAPPING_JSON  # type: ignore
else:
    from ..config import NODE_MAPPING_JSON  # type: ignore

from cadets.explanations import report_builder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = (
    REPO_ROOT
    / "cadets"
    / "artifact"
    / "explanations"
    / "2018-04-06_11_00_00~2018-04-06_12_15_00_explanations.json"
)


def _load_env(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip())
    return values


def main() -> None:
    env_vals = _load_env(REPO_ROOT / ".env")
    for key, value in env_vals.items():
        os.environ.setdefault(key, value)

    json_path = Path(os.environ.get("KAIROS_EXPLANATION_JSON", DEFAULT_JSON)).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Explanation JSON not found: {json_path}")

    mapping_path = Path(
        os.environ.get("KAIROS_NODE_MAPPING_JSON", NODE_MAPPING_JSON)
    ).resolve()
    if not mapping_path.exists():
        mapping_path = None

    data = json.loads(json_path.read_text())
    output_dir = (REPO_ROOT / "cadets" / "artifact" / "explanations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path, html_path, _ = report_builder.build_reports(
        data,
        output_dir,
        node_mapping_path=mapping_path,
        run_gpt=bool(os.environ.get("OPENAI_API_KEY")),
    )
    print("Markdown:", md_path)
    print("HTML:", html_path)


if __name__ == "__main__":
    main()
