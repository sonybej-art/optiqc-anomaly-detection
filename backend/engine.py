import io
import time
import base64
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import faiss
from scipy.ndimage import gaussian_filter


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
            urllib.request.urlretrieve(hf_url, checkpoint_path)
            print("Download complete.")

        print(f"Loading weights from {checkpoint_path}...")
        # weights_only=False is required here because the checkpoint bundles a
        # non-tensor object (the FAISS memory-bank index) alongside the state dict.
        # Only do this for checkpoints you trust / produced yourself.
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # WideResNet-50 backbone
        self.backbone = models.wide_resnet50_2(weights=None)
        self.backbone.load_state_dict(ckpt['feature_extractor_state'], strict=False)
        self.backbone.to(self.device).eval()

        # Faiss index & EVT thresholds
        self.index = ckpt['memory_bank_index']
        self.image_threshold = float(ckpt.get('image_threshold', 0.2036))
        self.pixel_threshold = float(ckpt.get('pixel_threshold', 0.2036))

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

    def _extract_patches(self, x: torch.Tensor):
        self.features = []
        with torch.no_grad():
            _ = self.backbone(x)
        f1, f2 = self.features[0], self.features[1]
        f1_resized = F.interpolate(f1, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f1_resized, f2], dim=1)

    def inspect_image(self, image_bytes: bytes, custom_threshold: float = None):
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 1. Feature extraction
        patches = self._extract_patches(tensor)
        B, C, H, W = patches.shape
        patch_vectors = patches.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy().astype(np.float32)

        # 2. Faiss nearest-neighbor search
        distances, _ = self.index.search(patch_vectors, 1)
        distances = np.sqrt(distances)

        # 3. Anomaly scoring & heatmap generation
        patch_scores = distances.reshape(H, W)
        image_score = float(np.max(patch_scores))

        smooth_map = gaussian_filter(patch_scores, sigma=4)
        map_tensor = torch.tensor(smooth_map).unsqueeze(0).unsqueeze(0)
        upscaled_map = F.interpolate(map_tensor, size=(256, 256), mode="bilinear", align_corners=False)
        anomaly_map = upscaled_map.squeeze().numpy()

        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        heatmap_uint8 = (norm_map * 255).astype(np.uint8)

        heatmap_img = Image.fromarray(heatmap_uint8).resize(img.size, Image.BILINEAR)
        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG")
        encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")

        threshold = custom_threshold if custom_threshold is not None else self.image_threshold
        is_defective = bool(image_score > threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "image_score": round(image_score, 4),
            "threshold": round(threshold, 4),
            "is_defective": is_defective,
            "defect_type": "STRUCTURAL_OR_SURFACE_ANOMALY" if is_defective else "NONE",
            "latency_ms": round(latency_ms, 2),
            "heatmap_base64": encoded_mask
        }
