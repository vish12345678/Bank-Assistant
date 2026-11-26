"""Central configuration. Values can be overridden in a .env file next to this one."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "kb"
INGESTED_DIR = KB_DIR / "ingested"
INDEX_DIR = ROOT / "index"
MODEL_CACHE = ROOT / ".model_cache"


def _load_dotenv() -> dict[str, str]:
    env_file = ROOT / ".env"
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


_DOTENV = _load_dotenv()


def setting(name: str, default: str | None = None) -> str | None:
    """.env file wins over process env, so the app does not pick up unrelated shell settings."""
    return _DOTENV.get(name) or os.environ.get(name) or default


ANTHROPIC_API_KEY = setting("ANTHROPIC_API_KEY")
# Only honour a base URL set explicitly for this app.
ANTHROPIC_BASE_URL = _DOTENV.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
MODEL = setting("ADVISOR_MODEL", "claude-opus-5-5")
EFFORT_EXTRACT = setting("ADVISOR_EFFORT_EXTRACT", "low")
EFFORT_ANSWER = setting("ADVISOR_EFFORT_ANSWER", "medium")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = int(setting("ADVISOR_TOP_K", "8") or 8)
PORT = int(setting("PORT", "8770") or 8770)

LLM_ENABLED = bool(ANTHROPIC_API_KEY)
