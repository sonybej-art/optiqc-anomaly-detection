import io
import time
import base64
import ctypes
import gc
import os
import shutil
import torch
import faiss
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import urllib.request
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter


torch.set_num_threads(1)
faiss.omp_set_num_threads(1)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, 0, 'constant')


class AnomalyInferenceEngine:
    def __init__(self, checkpoint_path: str = "weights/bottle_inspector_v2.pth", device: str = "cpu"):
        self.device = torch.device(device)

        checkpoint_directory = os.path.dirname(checkpoint_path) or "."
        os.makedirs(checkpoint_directory, exist_ok=True)

        if not os.path.exists(checkpoint_path):
            hf_url = "https://huggingface.co/javito3/optiqc-bottle-weights/resolve/main/bottle_inspector_v2.pth"
            print(f"Downloading checkpoint from {hf_url}...")
            with urllib.request.urlopen(hf_url) as response, open(checkpoint_path, "wb") as checkpoint_file:
                shutil.copyfileobj(response, checkpoint_file, length=1024 * 1024)
            print("Download complete.")

        print(f"Loading weights from {checkpoint_path}...")
        # weights_only=False is required here because the checkpoint bundles a
        # non-tensor object (the FAISS memory-bank index) alongside the state dict.
        # Only do this for checkpoints you trust / produced yourself.
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)

        # WideResNet-50 backbone
        self.backbone = models.wide_resnet50_2(weights=None)
        self.backbone.load_state_dict(ckpt['feature_extractor_state'], strict=False)
        self.backbone.to(self.device).eval()

        # Faiss index & EVT thresholds
        self.index = ckpt['memory_bank_index']
        total_vectors = getattr(self.index, "ntotal", None)
        dimension = getattr(self.index, "d", None)
        print(
            f"Memory bank loaded successfully. Vectors: {total_vectors}, Dim: {dimension}"
        )
        self.image_threshold = float(ckpt.get('image_threshold', 0.2036))
        self.pixel_threshold = float(ckpt.get('pixel_threshold', 0.2036))
        gc.collect()

        # Forward hooks for Layer 2 & Layer 3
        self.features = []
        def hook(module, input, output):
            self.features.append(output)
        self.backbone.layer2.register_forward_hook(hook)
        self.backbone.layer3.register_forward_hook(hook)

        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"Engine online. Calibrated EVT decision boundary: {self.image_threshold:.4f}")

    @torch.inference_mode()
    def _extract_patches(self, x: torch.Tensor):
        self.features = []
        _ = self.backbone(x)
        f1, f2 = self.features[0], self.features[1]
        f1_resized = F.interpolate(f1, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        patches = torch.cat([f1_resized, f2], dim=1)
        self.features.clear()
        del f1, f2, f1_resized
        return patches

    def _chunked_search(self, queries: np.ndarray, k: int = 1, chunk_size: int = 32) -> np.ndarray:
        n = queries.shape[0]
        out = np.empty((n, k), dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d, _ = self.index.search(queries[start:end], k)
            out[start:end] = d
        return out

    @torch.inference_mode()
    def inspect_image(self, image_bytes: bytes, custom_threshold: float = None):
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 1. Feature extraction
        patches = self._extract_patches(tensor)
        _, C, H, W = patches.shape
        patch_vectors = patches.permute(0, 2, 3, 1).reshape(-1, C)
        patch_vectors = patch_vectors.to(dtype=torch.float32).cpu().numpy()
        del tensor, patches

        # 2. Faiss nearest-neighbor search
        distances = self._chunked_search(patch_vectors, k=1, chunk_size=32)
        distances = np.sqrt(distances, dtype=np.float32)
        del patch_vectors

        # 3. Anomaly scoring & heatmap generation
        patch_scores = distances.reshape(H, W)
        image_score = float(np.max(patch_scores))

        smooth_map = gaussian_filter(patch_scores, sigma=4)
        del distances, patch_scores

        norm_map = (smooth_map - smooth_map.min()) / (smooth_map.max() - smooth_map.min() + 1e-8)
        map_uint8 = np.clip(norm_map * 255, 0, 255).astype(np.uint8)
        del smooth_map, norm_map

        resized_map = np.asarray(
            Image.fromarray(map_uint8).resize((256, 256), Image.BILINEAR),
            dtype=np.uint8,
        )
        del map_uint8

        colormap = np.empty((256, 3), dtype=np.uint8)
        colormap[:, 0] = np.minimum(255, resized_map.astype(np.uint16) * 2)
        colormap[:, 1] = np.minimum(255, resized_map.astype(np.uint16) * 2)
        colormap[:, 2] = 255 - resized_map
        heatmap_img = Image.fromarray(colormap[resized_map], mode="RGB")
        del resized_map, colormap

        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG")
        encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")
        del heatmap_img, buffer, img

        threshold = custom_threshold if custom_threshold is not None else self.image_threshold
        is_defective = bool(image_score > threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "image_score": round(image_score, 4),
            "threshold": round(threshold, 4),
            "is_defective": is_defective,
            "defect_type": "STRUCTURAL_OR_SURFACE_ANOMALY" if is_defective else "NONE",
            "latency_ms": round(latency_ms, 2),
            "heatmap_base64": encoded_mask
        }

        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return result
