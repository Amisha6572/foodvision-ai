"""
Ingredient lookup service.
Maps a predicted food label → ingredient list with quantities (grams).
"""

import json
import logging
import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/food_ingredients.json")


class IngredientService:
    """Loads the food→ingredient mapping and provides lookup methods."""

    def __init__(self):
        self._db: dict = {}
        self._load()

    def _load(self):
        path = os.path.abspath(DATA_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ingredient database not found at {path}")
        with open(path) as f:
            self._db = json.load(f)
        logger.info("Ingredient DB loaded: %d foods", len(self._db))

    def get_ingredients(self, food: str) -> Tuple[Dict[str, float], float]:
        """
        Return ingredient map and total weight for a food label.

        Args:
            food: food label string (e.g. "pizza")

        Returns:
            (ingredients_dict, total_weight_grams)

        Raises:
            KeyError: if food is not in the database.
        """
        food_key = food.lower().replace(" ", "_")
        if food_key not in self._db:
            raise KeyError(f"Food '{food}' not found in ingredient database.")

        ingredients = self._db[food_key]["ingredients"]
        total_weight = sum(ingredients.values())
        return ingredients, total_weight

    def get_ingredient_names(self, food: str) -> List[str]:
        """Return just the ingredient names for display."""
        ingredients, _ = self.get_ingredients(food)
        return list(ingredients.keys())

    @property
    def known_foods(self) -> List[str]:
        return list(self._db.keys())
