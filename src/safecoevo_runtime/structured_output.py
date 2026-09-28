"""Small, deterministic parsers for model-produced structured output."""
from __future__ import annotations

import json


def parse_single_json_object(raw: str) -> dict:
    """Accept a JSON object or one JSON markdown fence, and nothing else."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON markdown fence")
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value
