"""Generate Markdown reports for a given explanation JSON (GPT-powered)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

CAD_DIR = Path(__file__).resolve().parents[1]  # .../cadets
if str(CAD_DIR) not in sys.path:
    sys.path.insert(0, str(CAD_DIR))

import config  # type: ignore
from reporting import report_builder  # type: ignore

DEFAULT_JSON = CAD_DIR / "artifact" / "explanations" / "2018-04-06_11_00_00~2018-04-06_12_15_00_explanations.json"
DEFAULT_MAPPING_FALLBACK = CAD_DIR / "artifact" / "explanations" / "node_mapping.json"
DEFAULT_MAPPING = Path(getattr(config, "NODE_MAPPING_JSON", DEFAULT_MAPPING_FALLBACK))


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
    for key, value in _load_env(CAD_DIR / ".env").items():
        os.environ.setdefault(key, value)

    json_path = Path(os.environ.get("KAIROS_EXPLANATION_JSON", DEFAULT_JSON)).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Explanation JSON not found: {json_path}")

    mapping_path = Path(os.environ.get("KAIROS_NODE_MAPPING_JSON", DEFAULT_MAPPING)).resolve()
    if not mapping_path.exists():
        mapping_path = None

    data = json.loads(json_path.read_text())
    output_dir = (CAD_DIR / "artifact" / "explanations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpt = bool(os.environ.get("OPENAI_API_KEY"))
    existing_summary = data.get("gpt_summary")
    md_path, gpt_summary = report_builder.build_reports(
        data,
        output_dir,
        node_mapping_path=mapping_path,
        run_gpt=use_gpt,
        existing_summary=existing_summary,
    )

    if not md_path:
        raise RuntimeError("Markdown report was not generated.")

    if gpt_summary:
        data["gpt_summary"] = gpt_summary
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif existing_summary:
        print("[info] Reused existing GPT summary from artifact.")
    elif use_gpt:
        print("[warn] GPT summary unavailable; report exported without AI narrative.")
    else:
        print("[info] OPENAI_API_KEY not set; generated report without GPT narrative.")

    print("Markdown:", md_path)
    if gpt_summary:
        print("GPT summary:")
        print(gpt_summary)


if __name__ == "__main__":
    main()
