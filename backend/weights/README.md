# Model weights

This folder is where the trained checkpoints live at runtime. They are **not** committed to
this repo — GitHub hard-blocks any file over 100MB, and these checkpoints are ~267MB each.

Expected files:

- `bottle_inspector_v2.pth` — main fitted checkpoint
- `calibrated_inspector.pth` — EVT-calibrated boundary checkpoint
- `sensitive_inspector.pth` — high-sensitivity threshold checkpoint

## Recommended: host them on Hugging Face Hub

Hugging Face Hub is built for exactly this (free, no size drama, and it's the same place
you'd host a Space for a live demo later):

1. Create a model repo on https://huggingface.co/new (e.g. `your-username/optiqc-weights`).
2. Upload the three `.pth` files there via the web UI or `huggingface_hub`'s `upload_file`.
3. Before running the backend, download them into this folder — either manually, or by adding
   a small `huggingface_hub.hf_hub_download(...)` call at the top of `main.py` so it fetches
   them automatically if they're missing locally.
4. Link the model repo from the main README so anyone cloning this project can get the
   weights the same way.

## Alternative: Git LFS

If you'd rather keep everything in one GitHub repo, `git lfs track "*.pth"` before your first
commit works too — just note that GitHub's free LFS tier has a 1GB storage / 1GB bandwidth per
month cap, so it's fine for personal use but won't scale to many downloads.
