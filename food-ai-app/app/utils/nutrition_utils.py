"""
Nutrition helper utilities.
"""

from typing import Dict


def adjust_for_portion(nutrition: Dict[str, float], base_weight: float, portion: float) -> Dict[str, float]:
    """
    Scale nutrition values from base_weight to the requested portion size.

    Args:
        nutrition:   dict with keys calories, protein, carbs, fat (for base_weight grams)
        base_weight: total ingredient weight in grams for the base serving
        portion:     desired portion size in grams

    Returns:
        Scaled nutrition dict rounded to 1 decimal place.
    """
    if base_weight <= 0:
        return nutrition

    scale = portion / base_weight
    return {k: round(v * scale, 1) for k, v in nutrition.items()}


def compute_health_score(nutrition: Dict[str, float]) -> str:
    """
    Simple rule-based health score based on macro ratios.

    Scoring heuristic (per serving):
      - Calories < 300  → +2
      - Protein  > 15g  → +2
      - Fat      < 10g  → +1
      - Carbs    < 40g  → +1

    Score 5-6 → Healthy
    Score 3-4 → Moderate
    Score 0-2 → Indulgent
    """
    score = 0
    if nutrition.get("calories", 999) < 300:
        score += 2
    if nutrition.get("protein", 0) > 15:
        score += 2
    if nutrition.get("fat", 999) < 10:
        score += 1
    if nutrition.get("carbs", 999) < 40:
        score += 1

    if score >= 5:
        return "Healthy"
    elif score >= 3:
        return "Moderate"
    else:
        return "Indulgent"
