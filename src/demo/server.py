"""HTTP front for the demo pipeline.

POST /api/extract streams one JSON object per line (NDJSON) as each stage
finishes, so the page fills in while the VLM is still thinking.  Models load
once at startup; the pipeline runs one query at a time, because MPS is not safe
to share between threads.

    python src/demo/server.py                # live, http://127.0.0.1:8000
    python src/demo/server.py --replay       # stream saved runs of the examples, no models
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from demo.pipeline import ATTRIBUTES, ROOT, Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples.json")
WEB_DIST = os.path.join(ROOT, "demo_web", "dist")
IMAGES = os.path.join(ROOT, "data2024", "images")
_NAME = re.compile(r"^[\w\-+.]+\.(?:jpg|jpeg|png)$", re.I)

state: dict = {"pipeline": None, "replay": False, "load_s": None, "warm_ms": None}
lock = threading.Lock()


def load_examples() -> list[dict]:
    if not os.path.exists(EXAMPLES):
        return []
    with open(EXAMPLES) as fh:
        return json.load(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["examples"] = {e["id"]: e for e in load_examples()}
    if not state["replay"]:
        t = time.time()
        p = Pipeline()
        state["load_s"] = round(time.time() - t, 1)
        # The first MPS call compiles kernels; pay for it here, not in front of an audience.
        warm = next((e for e in state["examples"].values() if e["attribute"] in ("depth", "width", "height")), None)
        if warm:
            t = time.time()
            p.answer(warm["attribute"], image_path=os.path.join(IMAGES, warm["image_file"]))
            state["warm_ms"] = round(1000 * (time.time() - t))
        state["pipeline"] = p
    yield


app = FastAPI(title="Attribute extraction demo", lifespan=lifespan)


@app.get("/api/health")
def health():
    p = state["pipeline"]
    return {"mode": "replay" if state["replay"] else "live",
            "ready": state["replay"] or p is not None,
            "device": str(p.dev) if p else None,
            "vlm": bool(p and p.vlm is not None),
            "load_s": state["load_s"], "warmup_ms": state["warm_ms"],
            "attributes": list(ATTRIBUTES)}


@app.get("/api/examples")
def examples():
    return [{k: e[k] for k in ("id", "image_file", "attribute", "gold", "title", "note")}
            for e in state["examples"].values()]


@app.get("/api/images/{name}")
def image(name: str):
    if not _NAME.match(name) or not os.path.exists(os.path.join(IMAGES, name)):
        raise HTTPException(404)
    return FileResponse(os.path.join(IMAGES, name))


def _ndjson(events):
    for e in events:
        yield json.dumps(e) + "\n"


def _live(attribute, image_path, control, gold, cleanup):
    try:
        with lock:
            yield from state["pipeline"].run(attribute, image_path=image_path, control=control, gold=gold)
    except Exception as exc:                      # surface failures on the page, not only the console
        yield {"stage": "error", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if cleanup:
            os.unlink(image_path)


def _replay(example, control):
    events = example.get("events", {}).get(control)
    if not events:
        yield {"stage": "error", "message": f"no saved run of this example with control={control}"}
        return
    for e in events:
        time.sleep(min(e.get("ms", 0), 4000) / 1000)
        yield {**e, "replayed": True}


@app.post("/api/extract")
def extract(attribute: str = Form(...), example_id: str | None = Form(None),
            control: str = Form("real"), image: UploadFile | None = File(None)):
    if attribute not in ATTRIBUTES:
        raise HTTPException(422, f"attribute must be one of {', '.join(ATTRIBUTES)}")
    if control not in ("real", "blank"):
        raise HTTPException(422, "control must be 'real' or 'blank'")

    if example_id is not None:
        ex = state["examples"].get(example_id)
        if ex is None:
            raise HTTPException(404, "unknown example")
        if state["replay"]:
            if attribute != ex["attribute"]:
                raise HTTPException(409, "replay mode only has the example's own attribute")
            events = _replay(ex, control)
        else:
            gold = tuple(ex["gold"]) if attribute == ex["attribute"] else None
            events = _live(attribute, os.path.join(IMAGES, ex["image_file"]), control, gold, cleanup=False)
    elif image is not None:
        if state["replay"]:
            raise HTTPException(409, "uploads need live mode")
        # Phone photos carry their rotation as an EXIF tag, which the browser honours
        # and Vision does not; bake it in so the OCR boxes line up with what is shown.
        try:
            im = ImageOps.exif_transpose(Image.open(image.file)).convert("RGB")
        except Exception:
            raise HTTPException(415, "could not read that file as an image")
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        im.save(path, "JPEG", quality=95)
        events = _live(attribute, path, control, None, cleanup=True)
    else:
        raise HTTPException(422, "send an image or an example_id")

    return StreamingResponse(_ndjson(events), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="stream saved runs; loads no models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    state["replay"] = args.replay
    os.chdir(ROOT)                         # ocr.IMAGE_DIRS and the cache paths are repo-relative
    uvicorn.run(app, host=args.host, port=args.port)
