import logging
import os

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from engine import AnomalyInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optiqc")

app = FastAPI(
    title="OptiQC Anomaly Detection API",
    description="Edge industrial vision inspection pipeline.",
    version="1.0.0",
)

# Open CORS is intentional here — this is a public demo meant to be called
# from the browser-based index.html regardless of what origin serves it.
# If this ever backs a real product, swap allow_origins=["*"] for an
# explicit list of trusted frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "weights/bottle_inspector_v2.pth")

try:
    engine = AnomalyInferenceEngine(checkpoint_path=CHECKPOINT_PATH)
except FileNotFoundError:
    logger.error(
        "Checkpoint not found at '%s'. Make sure the weights file is present "
        "(or set CHECKPOINT_PATH) before the server can serve requests.",
        CHECKPOINT_PATH,
    )
    raise


@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    """Root health check — also used by index.html on load to confirm the
    backend is reachable and to pull the calibrated threshold."""
    return {
        "status": "healthy",
        "threshold": engine.image_threshold,
        "device": str(engine.device),
    }


@app.get("/health", include_in_schema=False)
def health_check_alias():
    """Same payload as '/', under the conventional '/health' path — some
    uptime monitors and orchestrators (k8s, HF Spaces readiness checks,
    etc.) expect this specific path by default."""
    return health_check()


@app.get("/demo", include_in_schema=False)
def serve_demo():
    """Serves the browser inspection panel (index.html)."""
    return FileResponse("index.html")


@app.post("/api/v1/inspect")
async def inspect(file: UploadFile = File(...), threshold: float = Form(None)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = engine.inspect_image(image_bytes=image_bytes, custom_threshold=threshold)
    except Exception as exc:
        # Don't leak a raw 500/stack trace to the client for a bad or
        # corrupted image — log the detail server-side, return a clean 400.
        logger.exception("Inspection failed for file '%s'", file.filename)
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    result["filename"] = file.filename
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)