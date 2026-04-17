# FoodVision AI – Complete Project Flow

## Overview

FoodVision AI is an end-to-end AI application that takes a food image as input and returns the food name, ingredients, nutritional values, and a health score — all adjustable by portion size.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                  Uploads food image + portion size              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND                             │
│                  streamlit_app/app.py                           │
│                                                                 │
│  • Image upload widget                                          │
│  • Portion size input                                           │
│  • Displays: food name, confidence, ingredients,                │
│    nutrition, health score, macro chart                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ POST /predict (multipart form)
                          │ { file: image, portion_size: float }
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                               │
│                   app/main.py                                   │
│                                                                 │
│  Endpoints:                                                     │
│  • POST /predict  → full prediction pipeline                    │
│  • GET  /foods    → list supported food labels                  │
│  • GET  /health   → service status                              │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────┐
│   FOOD       │  │  INGREDIENT   │  │   NUTRITION      │
│  CLASSIFIER  │  │   SERVICE     │  │    SERVICE       │
│              │  │               │  │                  │
│ HuggingFace  │  │ food_         │  │ ANN Model        │
│ nateraw/food │  │ ingredients   │  │ (Keras MLP)      │
│ ViT model    │  │ .json lookup  │  │ OR direct calc   │
│ (Food-101)   │  │               │  │ from nutrition   │
│ 101 classes  │  │ 101 foods ×   │  │ DB               │
│              │  │ ingredients   │  │                  │
└──────┬───────┘  └───────┬───────┘  └──────┬───────────┘
       │                  │                  │
       │   food label     │  ingredient map  │  nutrition values
       └──────────────────┴──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   PORTION SCALING     │
              │   + HEALTH SCORE      │
              │                       │
              │  adjust_for_portion() │
              │  compute_health_score │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   JSON RESPONSE       │
              │                       │
              │  food, confidence,    │
              │  ingredients,         │
              │  nutrition,           │
              │  adjusted_nutrition,  │
              │  health_score         │
              └───────────────────────┘
```

---

## Step-by-Step Data Flow

### Step 1 – Image Upload
- User uploads a food image (JPG/PNG/WEBP) via Streamlit
- Streamlit sends it as a `multipart/form-data` POST to `FastAPI /predict`
- Optional `portion_size` (grams) is sent alongside

### Step 2 – Image Preprocessing
```
app/utils/image_utils.py → preprocess_image()

Raw bytes → PIL Image → Resize to 224×224 → Normalize [0,1] → numpy array (1,224,224,3)
```

### Step 3 – Food Classification
```
app/services/food_classifier.py → FoodClassifier.predict()

Image → HuggingFace ViT (nateraw/food)
      → Top-5 Food-101 predictions
      → Pick highest-scoring label that exists in ingredient DB
      → Returns: ("Sushi", 0.94)
```

Model used: `nateraw/food` — ViT fine-tuned on Food-101 (101 food classes)

### Step 4 – Ingredient Lookup
```
app/services/ingredient_service.py → get_ingredients()

food_label → food_ingredients.json
           → { "rice": 100, "salmon": 60, "nori": 5, ... }
           → Returns ingredient map + total weight (grams)
```

### Step 5 – Nutrition Prediction
```
app/services/nutrition_service.py → predict_nutrition()

ingredient map → ANN Model (Keras MLP) if available
              OR direct calculation from ingredient_nutrition.json
              → { calories: 285, protein: 12, carbs: 36, fat: 10 }
```

ANN input: ingredient weight vector + ratios + total weight
ANN output: calories, protein, carbs, fat

### Step 6 – Portion Scaling
```
app/utils/nutrition_utils.py → adjust_for_portion()

base_nutrition × (portion_size / base_weight)
→ Scaled nutrition for user's requested portion
```

### Step 7 – Health Score
```
app/utils/nutrition_utils.py → compute_health_score()

Rules:
  calories < 300  → +2 pts
  protein  > 15g  → +2 pts
  fat      < 10g  → +1 pt
  carbs    < 40g  → +1 pt

Score 5-6 → Healthy  🥗
Score 3-4 → Moderate ⚖️
Score 0-2 → Indulgent 🍰
```

### Step 8 – Response
```json
{
  "food": "Sushi",
  "confidence": 0.94,
  "ingredients": ["rice", "salmon", "nori", "avocado", "cucumber"],
  "nutrition": { "calories": 285, "protein": 14, "carbs": 38, "fat": 8 },
  "portion_size": 200,
  "adjusted_nutrition": { "calories": 380, "protein": 18, "carbs": 50, "fat": 10 },
  "health_score": "Healthy",
  "base_weight": 215
}
```

---

## Project Structure

```
food-ai-app/
│
├── app/                          # FastAPI Backend
│   ├── main.py                   # API routes, startup, lifespan
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   ├── services/
│   │   ├── food_classifier.py    # HuggingFace ViT food detection
│   │   ├── ingredient_service.py # Food → ingredient lookup
│   │   └── nutrition_service.py  # ANN / direct nutrition calc
│   └── utils/
│       ├── image_utils.py        # Image decode + preprocess
│       └── nutrition_utils.py    # Portion scaling + health score
│
├── streamlit_app/
│   ├── app.py                    # Streamlit UI
│   ├── food_background.jpg       # Background image
│   ├── requirements.txt          # Frontend-only deps
│   └── .streamlit/
│       └── config.toml           # Theme config
│
├── training/
│   ├── train_cnn.py              # CNN training (Food11 / MobileNetV2)
│   ├── train_ann.py              # ANN training (nutrition predictor)
│   └── generate_mock_models.py   # Quick-start without training
│
├── models/                       # Saved model files
│   ├── food_cnn.h5               # Trained CNN (optional)
│   ├── nutrition_ann.h5          # Trained ANN
│   ├── class_labels.json         # CNN class index → label
│   ├── feature_names.json        # ANN input feature names
│   └── scaler.pkl                # ANN input scaler
│
├── data/
│   ├── food_ingredients.json     # 101 foods × ingredients (grams)
│   └── ingredient_nutrition.json # Per-gram macro values (80+ ingredients)
│
├── requirements.txt              # Full dependency list
├── Procfile                      # Render deployment config
└── README.md                     # Setup + deployment guide
```

---

## ML Models Used

| Model | Purpose | Source | Classes |
|-------|---------|--------|---------|
| `nateraw/food` ViT | Food classification | HuggingFace (Food-101) | 101 foods |
| Keras ANN (MLP) | Nutrition prediction | Trained locally | → 4 macros |
| MobileNetV2 | Fallback classifier | ImageNet (TF/Keras) | ~60 foods |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| ML Inference | HuggingFace Transformers (ViT) |
| ANN | TensorFlow / Keras |
| Data | JSON (ingredients + nutrition) |
| Deployment (BE) | Render / Railway |
| Deployment (FE) | Streamlit Cloud |

---

## Deployment Flow

```
GitHub Repo
    │
    ├──► Render (FastAPI backend)
    │    URL: https://foodvision-api.onrender.com
    │    Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    │
    └──► Streamlit Cloud (Frontend)
         URL: https://foodvision.streamlit.app
         File: food-ai-app/streamlit_app/app.py
         Secret: BACKEND_URL = https://foodvision-api.onrender.com
```

---

## Local Setup (Quick Start)

```bash
# 1. Install dependencies
conda activate ml_env
pip install -r requirements.txt

# 2. Generate mock models (skip if trained)
python training/generate_mock_models.py

# 3. Start backend (terminal 1)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Start frontend (terminal 2)
streamlit run streamlit_app/app.py
```

API docs: http://localhost:8000/docs
Frontend: http://localhost:8501
