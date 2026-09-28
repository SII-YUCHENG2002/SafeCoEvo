"""Prompt template loader for Safety Harness runtime agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPT_ROOT = Path(__file__).resolve().parent


def load_prompt_template(relative_path: str) -> str:
    """Load a packaged Markdown prompt template with a trailing newline."""

    path = PROMPT_ROOT / relative_path
    text = path.read_text(encoding="utf-8").rstrip()
    return text + "\n"


def render_prompt_template(relative_path: str, **values: Any) -> str:
    """Render named placeholders without using str.format, so JSON braces stay literal."""

    text = load_prompt_template(relative_path)
    for key, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        text = text.replace("{{" + key + "}}", value)
    return text
