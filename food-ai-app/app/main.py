"""
FoodVision AI – FastAPI Backend
Endpoints:
  POST /predict  – image → food label, ingredients, nutrition
  GET  /foods    – list all supported food labels
  GET  /health   – service health check
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models.schemas import ErrorResponse, PredictionResponse
from app.services.food_classifier import FoodClassifier
from app.services.ingredient_service import IngredientService
from app.services.nutrition_service import NutritionService
from app.utils.image_utils import preprocess_image
from app.utils.nutrition_utils import adjust_for_portion, compute_health_score

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Service singletons (loaded once at startup) ───────────────────────────────
classifier:  FoodClassifier   = None
ing_service: IngredientService = None
nut_service: NutritionService  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and data on startup."""
    global classifier, ing_service, nut_service
    logger.info("Loading services...")
    classifier  = FoodClassifier()
    ing_service = IngredientService()
    nut_service = NutritionService()
    logger.info("All services ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FoodVision AI",
    description="Food Detection, Ingredient & Nutrition Estimator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Returns service status."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/foods", tags=["Data"])
async def list_foods():
    """Returns all food labels supported by the system."""
    return {"foods": ing_service.known_foods}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="Predict food, ingredients, and nutrition from an image",
)
async def predict(
    file: UploadFile = File(..., description="Food image (JPEG/PNG)"),
    portion_size: Optional[float] = Form(
        default=None,
        description="Desired portion size in grams. Defaults to total ingredient weight.",
        gt=0,
        le=5000,
    ),
):
    """
    Upload a food image and receive:
    - Predicted food label + confidence
    - Ingredient list
    - Base nutrition values
    - Nutrition adjusted to requested portion size
    - Health score
    """
    # 1. Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPEG or PNG.",
        )

    # 2. Read and preprocess image
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image_array = preprocess_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 3. Classify food
    food_label, confidence = classifier.predict(image_array, image_bytes)
    logger.info("Predicted food: %s (%.2f%%)", food_label, confidence * 100)

    # 4. Get ingredients – map display label → DB key first
    db_key = classifier.get_db_key(food_label)
    try:
        ingredients, base_weight = ing_service.get_ingredients(db_key)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Food '{food_label}' was detected but has no ingredient data. "
                   "Try a different image.",
        )

    # 5. Predict nutrition (for base serving)
    nutrition = nut_service.predict_nutrition(ingredients)

    # 6. Adjust for portion size
    effective_portion = portion_size if portion_size else base_weight
    adjusted = adjust_for_portion(nutrition, base_weight, effective_portion)

    # 7. Health score (based on adjusted nutrition)
    health_score = compute_health_score(adjusted)

    return PredictionResponse(
        food=food_label,
        confidence=round(confidence, 4),
        ingredients=list(ingredients.keys()),
        nutrition=nutrition,
        portion_size=effective_portion,
        adjusted_nutrition=adjusted,
        health_score=health_score,
        base_weight=base_weight,
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
