"""Kodiak as a local HTTP decision service (Option B).

    python server.py                          # CPU, 8 threads, http://127.0.0.1:8765
    python server.py --device cuda            # use the GPU if you have one

POST /decide  {"state": ..., "questions": [...], "options": {...}}
          ->  {"model": ..., "latency_ms": <forward pass>, "answers": {question id: answer}}
GET  /health  -> {"status": "ok", "model": ..., "device": ...}

The request format is Kodiak's own Request (the same one the model repo's handler.py takes inside "inputs"), and
{"inputs": {...}} is accepted too. The model is loaded once at startup. The server binds to localhost only.
"""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from kodiak_s1.hub import Kodiak

DEFAULT_MODEL = "cortex-agent-llc/kodiak-small-r1-preview"


def create_app(model: str = DEFAULT_MODEL, device: str = "cpu", threads: int = 8) -> FastAPI:
    state: dict[str, Any] = {}
    lock = threading.Lock()  # one forward pass at a time; torch already uses every thread we give it

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if threads > 0:
            torch.set_num_threads(threads)
        t0 = time.perf_counter()
        state["kodiak"] = Kodiak.from_pretrained(model, device=device)
        state["device"] = str(next(state["kodiak"].model.parameters()).device)
        # Warm up (first calls pay for lazy init and kernel selection).
        for _ in range(2):
            state["kodiak"].decide("warm-up", [{"type": "choice", "id": "w", "text": "Warm up?", "labels": ["yes", "no"]}])
        print(f"Kodiak {model} ready on {state['device']} in {time.perf_counter() - t0:.1f}s", flush=True)
        yield
        state.clear()

    app = FastAPI(title="Kodiak decision service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model, "device": state.get("device")}

    @app.post("/decide")
    def decide(body: dict[str, Any]) -> dict:  # sync handler: runs in the threadpool, off the event loop
        req = body.get("inputs", body)
        if not isinstance(req, dict) or "state" not in req or "questions" not in req:
            raise HTTPException(422, "expected {'state': ..., 'questions': [...], 'options': {...}}")
        req = {k: req[k] for k in ("state", "questions", "options") if k in req}
        try:
            with lock:
                return state["kodiak"].answer([req])[0]
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False)) from e

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or a local model folder")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or auto (default: cpu)")
    ap.add_argument("--threads", type=int, default=8,
                    help="torch CPU threads (default 8; more is not always faster, 0 = torch default)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    # Localhost only: this is a demo service with no auth.
    uvicorn.run(create_app(a.model, a.device, a.threads), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
