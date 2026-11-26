"""FastAPI server: serves the chat UI and streams advisor events as Server-Sent Events."""
from __future__ import annotations

import json
from collections import Counter

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import config
from advisor import Advisor

app = FastAPI(title="Bank Product Advisor")
advisor = Advisor()


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []
    bank: str | None = None


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    r = advisor.retriever
    return {
        "llm": config.LLM_ENABLED,
        "model": config.MODEL if config.LLM_ENABLED else None,
        "retrieval": r.mode,
        "products": len(r.products),
        "categories": Counter(p.category for p in r.products),
        "banks": ["All banks", *r.banks()],
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    history = [t.model_dump() for t in req.history if t.content.strip()]

    def events():
        try:
            for ev in advisor.run(req.message, history, req.bank):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # surface errors in the UI instead of a dead stream
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=config.PORT)
