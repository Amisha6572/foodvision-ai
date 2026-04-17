"""
Nutrition prediction service.
Uses the trained ANN when available; falls back to direct calculation
from the ingredient nutrition database.
"""

import json
import logging
import os
import pickle
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

NUTRITION_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/ingredient_nutrition.json")
ANN_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "../../models/nutrition_ann.h5")
SCALER_PATH       = os.path.join(os.path.dirname(__file__), "../../models/scaler.pkl")
FEATURES_PATH     = os.path.join(os.path.dirname(__file__), "../../models/feature_names.json")


class NutritionService:
    """Predicts nutrition values from an ingredient map."""

    def __init__(self):
        self._nutrition_db: dict = {}
        self._ann_model = None
        self._scaler = None
        self._feature_names: list = []
        self._load()

    def _load(self):
        # Always load the nutrition DB (used as fallback)
        db_path = os.path.abspath(NUTRITION_DB_PATH)
        with open(db_path) as f:
            self._nutrition_db = json.load(f)

        # Try loading ANN model + scaler
        ann_path    = os.path.abspath(ANN_MODEL_PATH)
        scaler_path = os.path.abspath(SCALER_PATH)
        feat_path   = os.path.abspath(FEATURES_PATH)

        if os.path.exists(ann_path) and os.path.exists(scaler_path) and os.path.exists(feat_path):
            try:
                import tensorflow as tf
                # compile=False avoids metric deserialization issues across Keras versions
                self._ann_model = tf.keras.models.load_model(ann_path, compile=False)
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                with open(feat_path) as f:
                    self._feature_names = json.load(f)
                logger.info("ANN nutrition model loaded.")
            except Exception as exc:
                logger.warning("Could not load ANN model: %s – using direct calculation.", exc)
        else:
            logger.info("ANN model not found – using direct nutrition calculation.")

    def _direct_calculate(self, ingredients: Dict[str, float]) -> Dict[str, float]:
        """Calculate nutrition directly from ingredient DB (per-gram values × grams)."""
        totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for ing, grams in ingredients.items():
            if ing in self._nutrition_db:
                n = self._nutrition_db[ing]
                totals["calories"] += n["calories"] * grams
                totals["protein"]  += n["protein"]  * grams
                totals["carbs"]    += n["carbs"]     * grams
                totals["fat"]      += n["fat"]       * grams
        return {k: round(v, 1) for k, v in totals.items()}

    def _ann_predict(self, ingredients: Dict[str, float]) -> Dict[str, float]:
        """Use the ANN model to predict nutrition."""
        total_weight = sum(ingredients.values())
        feature_vec = []
        for ing in self._feature_names:
            grams = ingredients.get(ing, 0.0)
            ratio = grams / total_weight if total_weight > 0 else 0.0
            feature_vec.extend([grams, ratio])
        feature_vec.append(total_weight)

        X = np.array([feature_vec], dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        pred = self._ann_model.predict(X_scaled, verbose=0)[0]

        return {
            "calories": round(float(max(pred[0], 0)), 1),
            "protein":  round(float(max(pred[1], 0)), 1),
            "carbs":    round(float(max(pred[2], 0)), 1),
            "fat":      round(float(max(pred[3], 0)), 1),
        }

    def predict_nutrition(self, ingredients: Dict[str, float]) -> Dict[str, float]:
        """
        Predict nutrition for a given ingredient map.
        Uses ANN if available, otherwise direct calculation.
        """
        if self._ann_model is not None:
            try:
                return self._ann_predict(ingredients)
            except Exception as exc:
                logger.warning("ANN prediction failed: %s – falling back.", exc)

        return self._direct_calculate(ingredients)
