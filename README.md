# OptiQC — CPU-Efficient Anomaly Detection for Industrial Visual Inspection

A few-shot, CPU-only visual anomaly detection system for industrial quality control. Trained and calibrated as a research project, then packaged into a REST API and an interactive inspection demo so the results are something you can actually click through, not just read.

<!-- Add your recorded demo here once you have it: ![demo](demo/demo.gif) -->

## Overview

Manual visual inspection on a production line is slow and inconsistent, and most anomaly-detection research assumes GPU inference and large labeled defect datasets — neither of which is realistic at the edge. This project targets the opposite constraints: **few-shot** (10-shot), **CPU-only**, and still fast enough for inline inspection.

The approach is a PatchCore-style memory-bank detector: a WideResNet-50 backbone extracts mid-level features (layers 2 and 3), a FAISS index holds a memory bank of "normal" patch embeddings, and anomaly scores come from nearest-neighbor distance in that embedding space. A decision boundary is calibrated using Extreme Value Theory (EVT) rather than a naive percentile cutoff, and 8-bit feature quantization is used to cut retrieval latency and memory footprint without materially hurting detection accuracy.

## Results

| Metric | Value |
|---|---|
| Image AUROC (5-seed mean, 10-shot) | 0.7792 |
| Retrieval speedup (8-bit quantized) | 51.7x → 12.7 ms |
| Memory footprint reduction | 11.8x → 38 MB |
| Benchmark | MVTec AD |

Full methodology and the quantization–EVT coupling analysis are in the research report — see [`research/`](research/).

## Architecture

```
image
  │
  ▼
SquarePad + resize (256×256)
  │
  ▼
WideResNet-50  ── forward hooks on layer2 + layer3
  │
  ▼
patch feature map  (concatenated, upsampled)
  │
  ▼
FAISS memory-bank nearest-neighbor search
  │
  ▼
patch anomaly map ── Gaussian smoothing ── upscale
  │
  ▼
image-level score  vs.  EVT-calibrated threshold
  │
  ▼
PASS / FAIL  +  anomaly heatmap
```

## Repo structure

```
optiqc-anomaly-detection/
├── backend/              FastAPI inference service
│   ├── engine.py         model loading + inference logic
│   ├── main.py           API routes
│   ├── requirements.txt
│   └── weights/          checkpoints go here — see weights/README.md (not committed to git)
├── demo/                 self-contained HTML inspection UI
│   └── optiqc_inspector_demo.html
├── research/             training code + report
│   └── README.md         what to drop in here
└── README.md
```

## Running it locally

1. Clone the repo.
2. Get the checkpoint files into `backend/weights/` — see [`backend/weights/README.md`](backend/weights/README.md).
3. ```
   cd backend
   pip install -r requirements.txt
   python main.py
   ```
4. Open `demo/optiqc_inspector_demo.html` directly in your browser (double-click it — don't serve it over https, to avoid mixed-content issues with the local API). Confirm the API endpoint field reads `http://localhost:8000`, upload an image, and run an inspection.
5. Interactive API docs are also available at `http://localhost:8000/docs`.

## Tech stack

PyTorch · torchvision (WideResNet-50) · FAISS · FastAPI · Extreme Value Theory calibration · vanilla JS/HTML/Canvas frontend

## License

MIT — see [LICENSE](LICENSE).
