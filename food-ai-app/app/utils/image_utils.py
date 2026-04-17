"""
Image preprocessing utilities for the CNN food classifier.
"""

import io
import logging
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMG_SIZE = 224  # MobileNetV2 expected input


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes, resize to (224, 224), and normalise to [0, 1].

    Returns:
        np.ndarray of shape (1, 224, 224, 3) ready for model inference.

    Raises:
        ValueError: if the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        logger.error("Failed to decode image: %s", exc)
        raise ValueError(f"Invalid image data: {exc}") from exc

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)
