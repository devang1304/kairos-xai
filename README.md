# Kairos XAI (CADETS)

Minimal instructions to set up the environment, run the Kairos pipeline, generate explanations, and view analyst‑friendly reports and charts.

---

## 1) Environment Setup

Requirements
- Python 3.10+
- numpy < 2.0 (pinned in requirements.txt)
- PostgreSQL for node mapping export (optional; dashboard works without mapping)

Create environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install core dependencies (reporting + dashboard + exporter)
pip install -r requirements.txt

# Install PyTorch + PyG (choose ONE of the following)

# Option A: CPU‑only (simple)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Option B: CUDA 12.1 (example; adjust for your CUDA)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Verify
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
PY
```

Project config
- Edit `cadets/config.py` and set:
  - `RAW_DIR` to your CADETS data root (if running the full pipeline)
  - DB settings if you plan to export the node mapping from PostgreSQL

---

## 2) Running the Kairos Pipeline (caution: training is heavy)

Recommended approach is to run individual Makefile targets from `cadets/` rather than a single all‑in‑one. Training is heavy; avoid launching it accidentally.
```bash
cd cadets

# Prepare embeddings / train / test (choose only what you need)
make embeddings   # vectorize
make train        # heavy
make test         # evaluate

# Anomaly workflow (no training)
make anomalous_queue
make evaluation
make attack_investigation
```

Note: avoid accidentally running `make pipeline` on large datasets unless you intend to train. Use the individual targets above for reproducibility and quicker iteration.

Artifacts land under `cadets/artifact/`.

---

## 3) Explanations Pipeline

Generates window‑based explanation JSON with GraphMask + VA‑TG highlights under `artifact/explanations/`:
```bash
cd cadets
python -m explanations.window_analysis
```

Optional: ensure node mapping (labels) exists. The pipeline will try to create it automatically if it’s missing, using DB settings from `config.py`. You can also run it manually:
```bash
python -m explanations.export_node_mapping
```

Outputs include (example):
- `artifact/explanations/2018-04-06_11_00_00~2018-04-06_12_15_00_explanations.json`
- `artifact/explanations/graph_4_6_summary.json`
- `artifact/explanations/temporal_explanations.log`

---

## 4) Reporting + Dashboard

Generate a Markdown report with optional GPT narrative (reused between runs):
```bash
cd cadets
export KAIROS_EXPLANATION_JSON="artifact/explanations/2018-04-06_11_00_00~2018-04-06_12_15_00_explanations.json"
export KAIROS_NODE_MAPPING_JSON="artifact/explanations/node_mapping.json"   # optional
# export OPENAI_API_KEY=sk-...                                             # optional GPT

python -m reporting.generate_report
```

Launch the dashboard (top edges, top nodes with threshold line, and timeline + colocated GPT sections):
```bash
streamlit run reporting/streamlit_dashboard.py
```

What you’ll see
- Bar chart of top GraphMask edges (relation‑colored)
- Bar chart of top nodes by average loss with a dashed event‑loss threshold line
- Minute‑binned event timeline
- GPT narrative sections (What happened? Who’s involved? Why flagged?) aligned next to the relevant charts

---

## 5) Logging & Progress

The codebase favors minimal prints and visible progress where useful:
- Explanations write to `artifact/explanations/temporal_explanations.log`.
- Console output uses a few concise prints and `tqdm` progress bars for streaming/explainer loops.
- GPU memory checks are printed only when relevant (OOM fallback, low‑memory warnings).

Tip: for long runs, tail the log:
```bash
tail -f cadets/artifact/explanations/temporal_explanations.log
```

---

## 6) Environment Variables (optional)

- `KAIROS_EXPLANATION_JSON` – path to a single explanation JSON for report/dashboard
- `KAIROS_NODE_MAPPING_JSON` – mapping JSON (labels), if available
- `OPENAI_API_KEY` – enables GPT narrative in report (dashboard reuses saved summaries)

---

## 7) Notes

- Threshold semantics: the event‑loss threshold is derived from per‑event losses and is used to check whether any node participates in an event above the cut‑off (peak loss). The dashboard’s “Top nodes” chart shows average loss per node and draws a dashed vertical line to indicate the event‑loss threshold context.
- The dashboard is pandas‑free for reliability; visuals use Plotly with plain Python lists/dicts.
