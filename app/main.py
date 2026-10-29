"""Web server: the static site plus a JSON API around :class:`engine.Engine`.

Run locally:  python -m uvicorn main:app --app-dir app --port 7860
Endpoints:    GET /            the website
              GET /api/health  {"ready": bool, ...}
              POST /api/describe  multipart: image (file), mode (safe|strict|forced)
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path

# Local machine: keep Hugging Face downloads on D: (C: is nearly full). Hosts set their own.
if "HF_HOME" not in os.environ and Path("D:/hf_cache").exists():
    os.environ["HF_HOME"] = "D:/hf_cache"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from engine import CONFIG, MODES, Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Grounded Product Descriptions", docs_url="/api/docs")
engine = Engine()


def _load() -> None:
    try:
        engine.load()
    except Exception as exc:  # surfaced by /api/health instead of crashing the site
        logging.exception("model loading failed")
        engine.error = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def startup() -> None:
    threading.Thread(target=_load, daemon=True).start()  # site is up while models load


@app.get("/api/health")
def health() -> dict:
    return {"ready": engine.ready, "error": engine.error,
            "extractor": CONFIG["extractor"], "llm": getattr(engine, "llm_name", CONFIG["llm"]),
            "device": getattr(engine, "device", None)}


@app.post("/api/describe")
async def describe(image: UploadFile = File(...), mode: str = Form("safe")) -> JSONResponse:
    if not engine.ready:
        raise HTTPException(503, engine.error or "Models are still loading, try again in a minute.")
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
    data = await image.read()
    if len(data) > CONFIG["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {CONFIG['max_upload_mb']} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not an image we can read (use JPG, PNG or WEBP).")
    result = await run_in_threadpool(engine.describe, img, mode)  # keep the event loop free
    return JSONResponse(result.as_dict())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
