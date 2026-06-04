"""Model save/load without ML training dependencies (safe for Streamlit Cloud)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


def save_model(model: Any, path: Path, metadata: Optional[dict] = None) -> None:
    """Save model with optional metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, path)
    logger.info("Saved model to %s", path)


def load_model(path: Path) -> Tuple[Any, dict]:
    """Load model and metadata."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})
    return payload, {}
