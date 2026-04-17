"""
Food classification service using the trained CNN (MobileNetV2).
Falls back to a rule-based mock when the model file is absent (dev mode).
"""

import json
import logging
import os
import random
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../../models/food_cnn.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "../../models/class_labels.json")

# Food11 category labels (matches FOOD11_LABELS in train_cnn.py)
KNOWN_FOODS = [
    "bread",
    "dairy_product",
    "dessert",
    "egg",
    "fried_food",
    "meat",
    "noodles_pasta",
    "rice",
    "seafood",
    "soup",
    "vegetable_fruit",
]


class FoodClassifier:
    """Wraps the CNN model for food label prediction."""

    def __init__(self):
        self._model = None
        self._labels: dict = {}
        self._load()

    def _load(self):
        model_path = os.path.abspath(MODEL_PATH)
        labels_path = os.path.abspath(LABELS_PATH)

        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(model_path)
                logger.info("CNN model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load CNN model: %s – using mock.", exc)

        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self._labels = json.load(f)
        else:
            # Build label map from known foods
            self._labels = {str(i): food for i, food in enumerate(KNOWN_FOODS)}

    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """
        Predict food label from a preprocessed image array (1, 224, 224, 3).

        Returns:
            (food_label, confidence) tuple.
        """
        if self._model is not None:
            preds = self._model.predict(image_array, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            label = self._labels.get(str(idx), "unknown")
            return label, round(confidence, 4)

        # ── Mock mode (no trained model yet) ──────────────────────────────
        logger.warning("Running in MOCK mode – no trained CNN model found.")
        label = random.choice(KNOWN_FOODS)
        confidence = round(random.uniform(0.70, 0.97), 4)
        return label, confidence
