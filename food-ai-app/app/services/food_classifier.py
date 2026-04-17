"""
Food classification service.

Uses the Hugging Face `nateraw/food` pipeline — a ViT model fine-tuned on
Food-101 (101 classes, ~90% accuracy). Downloads ~330MB on first run and
caches locally. No training required.

Fallback: MobileNetV2 ImageNet decode (limited food coverage).
"""

import logging
import os
import random
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Food-101 class list (all 101 labels) ─────────────────────────────────────
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

# ── Map Food-101 label → ingredient DB key ────────────────────────────────────
# Labels that map directly use the same name; others map to closest entry
FOOD101_TO_DB = {
    "apple_pie":              "apple_pie",
    "baby_back_ribs":         "baby_back_ribs",
    "baklava":                "baklava",
    "beef_carpaccio":         "beef_carpaccio",
    "beef_tartare":           "beef_tartare",
    "beet_salad":             "beet_salad",
    "beignets":               "beignets",
    "bibimbap":               "bibimbap",
    "bread_pudding":          "bread_pudding",
    "breakfast_burrito":      "burrito",
    "bruschetta":             "bruschetta",
    "caesar_salad":           "caesar_salad",
    "cannoli":                "cannoli",
    "caprese_salad":          "caprese_salad",
    "carrot_cake":            "carrot_cake",
    "ceviche":                "ceviche",
    "cheesecake":             "cheesecake",
    "cheese_plate":           "cheese_plate",
    "chicken_curry":          "chicken_curry",
    "chicken_quesadilla":     "chicken_quesadilla",
    "chicken_wings":          "chicken_wings",
    "chocolate_cake":         "chocolate_cake",
    "chocolate_mousse":       "chocolate_mousse",
    "churros":                "churros",
    "clam_chowder":           "clam_chowder",
    "club_sandwich":          "club_sandwich",
    "crab_cakes":             "crab_cakes",
    "creme_brulee":           "creme_brulee",
    "croque_madame":          "croque_madame",
    "cup_cakes":              "cup_cakes",
    "deviled_eggs":           "deviled_eggs",
    "donuts":                 "donuts",
    "dumplings":              "dumplings",
    "edamame":                "edamame",
    "eggs_benedict":          "eggs_benedict",
    "escargots":              "escargots",
    "falafel":                "falafel",
    "filet_mignon":           "filet_mignon",
    "fish_and_chips":         "fish_and_chips",
    "foie_gras":              "foie_gras",
    "french_fries":           "french_fries",
    "french_onion_soup":      "onion_soup",
    "french_toast":           "french_toast",
    "fried_calamari":         "fried_calamari",
    "fried_rice":             "fried_rice",
    "frozen_yogurt":          "frozen_yogurt",
    "garlic_bread":           "garlic_bread",
    "gnocchi":                "gnocchi",
    "greek_salad":            "greek_salad",
    "grilled_cheese_sandwich":"grilled_cheese_sandwich",
    "grilled_salmon":         "grilled_salmon",
    "guacamole":              "guacamole",
    "gyoza":                  "gyoza",
    "hamburger":              "hamburger",
    "hot_and_sour_soup":      "hot_and_sour_soup",
    "hot_dog":                "hotdog",
    "huevos_rancheros":       "huevos_rancheros",
    "hummus":                 "hummus",
    "ice_cream":              "ice_cream",
    "lasagna":                "lasagna",
    "lobster_bisque":         "lobster_bisque",
    "lobster_roll_sandwich":  "lobster_roll",
    "macaroni_and_cheese":    "mac_and_cheese",
    "macarons":               "macarons",
    "miso_soup":              "miso_soup",
    "mussels":                "mussels",
    "nachos":                 "nachos",
    "omelette":               "omelette",
    "onion_rings":            "onion_rings",
    "oysters":                "oysters",
    "pad_thai":               "pad_thai",
    "paella":                 "paella",
    "pancakes":               "pancakes",
    "panna_cotta":            "panna_cotta",
    "peking_duck":            "peking_duck",
    "pho":                    "pho",
    "pizza":                  "pizza",
    "pork_chop":              "pork_chop",
    "poutine":                "poutine",
    "prime_rib":              "prime_rib",
    "pulled_pork_sandwich":   "pulled_pork_sandwich",
    "ramen":                  "ramen",
    "ravioli":                "ravioli",
    "red_velvet_cake":        "red_velvet_cake",
    "risotto":                "risotto",
    "samosa":                 "samosa",
    "sashimi":                "sashimi",
    "scallops":               "scallops",
    "seaweed_salad":          "seaweed_salad",
    "shrimp_and_grits":       "shrimp_and_grits",
    "spaghetti_bolognese":    "spaghetti_bolognese",
    "spaghetti_carbonara":    "carbonara",
    "spring_rolls":           "spring_roll",
    "steak":                  "steak",
    "strawberry_shortcake":   "strawberry_shortcake",
    "sushi":                  "sushi",
    "tacos":                  "tacos",
    "takoyaki":               "takoyaki",
    "tiramisu":               "tiramisu",
    "tuna_tartare":           "tuna_tartare",
    "waffles":                "waffle",
}


class FoodClassifier:
    """
    Uses Hugging Face `nateraw/food` ViT model (Food-101, 101 classes).
    Falls back to MobileNetV2 ImageNet if transformers not installed.
    """

    def __init__(self):
        self._pipeline = None
        self._fallback_model = None
        self._load()

    def _load(self):
        # Try Hugging Face pipeline first
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "image-classification",
                model="nateraw/food",
                top_k=5,
            )
            logger.info("HuggingFace Food-101 model (nateraw/food) loaded.")
            return
        except Exception as exc:
            logger.warning("HuggingFace pipeline failed: %s", exc)

        # Fallback: MobileNetV2 ImageNet
        try:
            import tensorflow as tf
            self._fallback_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), include_top=True, weights="imagenet"
            )
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            self._fallback_model.predict(dummy, verbose=0)
            logger.info("Fallback: MobileNetV2 ImageNet loaded.")
        except Exception as exc:
            logger.warning("Fallback model also failed: %s", exc)

    def predict(self, image_array: np.ndarray, image_bytes: bytes = None) -> Tuple[str, float]:
        """
        Returns (display_label, confidence).
        Tries HuggingFace Food-101 pipeline first, then MobileNetV2 fallback.
        """
        # ── HuggingFace Food-101 pipeline ─────────────────────────────────
        if self._pipeline is not None and image_bytes is not None:
            try:
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                results = self._pipeline(img)

                # results = [{"label": "sushi", "score": 0.95}, ...]
                # Pick top result that has a DB mapping
                for r in results:
                    label = r["label"].lower().replace(" ", "_")
                    if label in FOOD101_TO_DB:
                        display = label.replace("_", " ").title()
                        logger.info("Food-101 prediction: %s (%.1f%%)", display, r["score"] * 100)
                        return display, round(float(r["score"]), 4)

                # Use top result regardless
                label = results[0]["label"].lower().replace(" ", "_")
                display = label.replace("_", " ").title()
                return display, round(float(results[0]["score"]), 4)

            except Exception as exc:
                logger.warning("HuggingFace inference failed: %s – trying fallback.", exc)

        # ── MobileNetV2 fallback ──────────────────────────────────────────
        if self._fallback_model is not None:
            import tensorflow as tf
            img = image_array * 2.0 - 1.0
            preds = self._fallback_model.predict(img, verbose=0)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=20)[0]

            # Pick first label that maps to a known food
            IMAGENET_FOOD = {
                "pizza","cheeseburger","hamburger","hotdog","French_fries",
                "fried_chicken","spring_roll","burrito","carbonara","spaghetti_squash",
                "meat_loaf","steak","pork_loin","pot_roast","sushi","lobster_bisque",
                "crab_cakes","consomme","clam_chowder","onion_soup","ice_cream",
                "chocolate_cake","trifle","ice_lolly","eggnog","guacamole","potpie",
                "strawberry","banana","orange","lemon","pineapple","pomegranate",
                "fig","jackfruit","broccoli","cauliflower","corn","artichoke",
                "cucumber","bell_pepper","mushroom","bagel","pretzel","croissant",
                "French_loaf","waffle","rooster","hen",
            }
            for imagenet_label, human_label, score in decoded:
                if imagenet_label in IMAGENET_FOOD:
                    display = human_label.replace("_", " ").title()
                    logger.info("MobileNetV2 fallback: %s (%.1f%%)", display, score * 100)
                    return display, round(float(score), 4)

            _, human_label, score = decoded[0]
            return human_label.replace("_", " ").title(), round(float(score), 4)

        # ── Mock ──────────────────────────────────────────────────────────
        label = random.choice(FOOD101_CLASSES)
        return label.replace("_", " ").title(), round(random.uniform(0.65, 0.95), 4)

    def get_db_key(self, display_label: str) -> str:
        """Map display label → ingredient DB key."""
        normalized = display_label.lower().replace(" ", "_")
        # Direct match in Food-101 map
        if normalized in FOOD101_TO_DB:
            return FOOD101_TO_DB[normalized]
        # Partial match
        for food101_key, db_key in FOOD101_TO_DB.items():
            if food101_key in normalized or normalized in food101_key:
                return db_key
        logger.warning("No DB key for '%s', defaulting to pizza", display_label)
        return "pizza"
