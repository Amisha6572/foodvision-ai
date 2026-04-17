"""
Pydantic schemas for request/response validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class NutritionInfo(BaseModel):
    calories: float = Field(..., description="Calories in kcal")
    protein:  float = Field(..., description="Protein in grams")
    carbs:    float = Field(..., description="Carbohydrates in grams")
    fat:      float = Field(..., description="Fat in grams")


class PredictionResponse(BaseModel):
    food:               str
    confidence:         float
    ingredients:        List[str]
    nutrition:          NutritionInfo
    portion_size:       float = Field(..., description="Portion size used (grams)")
    adjusted_nutrition: NutritionInfo
    health_score:       str
    base_weight:        float = Field(..., description="Total ingredient weight for base serving (grams)")


class ErrorResponse(BaseModel):
    detail: str
