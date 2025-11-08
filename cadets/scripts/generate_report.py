"""Generate Markdown/HTML reports for a given explanation JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    os.sys.path.append(str(REPO_ROOT))
    from cadets.explanations import report_builder  # type: ignore
else:
    from cadets.explanations import report_builder

CONFIG_PATH = REPO_ROOT / "cadets" / "config.py"
DEFAULT_JSON = (
    REPO_ROOT
    / "cadets"
    / "artifact"
    / "explanations"
    / "2018-04-06_11_00_00~2018-04-06_12_15_00_explanations.json"
)

def _load_config() -> Dict[str, str]:
    namespace: Dict[str, object] = {}
    exec(CONFIG_PATH.read_text(encoding="utf-8"), namespace)
    return namespace  # type: ignore[return-value]


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
    for key, value in _load_env(REPO_ROOT / ".env").items():
        os.environ.setdefault(key, value)
    config = _load_config()

    json_path = Path(os.environ.get("KAIROS_EXPLANATION_JSON", DEFAULT_JSON)).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Explanation JSON not found: {json_path}")

    mapping_default = config.get("NODE_MAPPING_JSON", DEFAULT_JSON.parent / "node_mapping.json")
    mapping_path = Path(os.environ.get("KAIROS_NODE_MAPPING_JSON", mapping_default)).resolve()
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
