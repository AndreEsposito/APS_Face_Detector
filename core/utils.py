from __future__ import annotations
import logging
import time
from pathlib import Path
import cv2
import numpy as np
from .paths import CAPTURES_DIR

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

def put_text(frame: np.ndarray, text: str, org, scale=0.6, color=(0, 255, 0), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def save_frame(frame: np.ndarray, out_dir: Path = CAPTURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"frame_{int(time.time())}.jpg"
    cv2.imwrite(str(out), frame)
    logging.info("Frame salvo em: %s", out)
    return out
