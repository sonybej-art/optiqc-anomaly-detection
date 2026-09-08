# Research

This is where the training pipeline and write-up live — arguably the most important folder
in the repo, since it's the actual research contribution behind the deployed API.

Drop in:

- **`train.ipynb`** — export your Colab notebook (File → Download → Download .ipynb) and add
  it here. If it depends on a different environment than the backend, add a
  `requirements.txt` alongside it.
- **`report.pdf`** (or `.md`) — the write-up on quantization–EVT coupling in few-shot,
  CPU-constrained anomaly detection. Once it's here, link it from the main README's Results
  section — this is the piece that separates "trained a model" from "conducted research."
- Any evaluation scripts / plots used to produce the AUROC, latency, and memory numbers
  quoted in the main README, so the results are reproducible from this repo alone.
