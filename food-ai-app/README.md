# 🍽️ FoodVision AI

**Food Detection · Ingredient Analysis · Nutrition Estimator**

A production-level end-to-end AI application that classifies food from images, identifies ingredients, and estimates nutritional values with portion-size adjustment.

---

## Project Structure

```
food-ai-app/
├── app/                        # FastAPI backend
│   ├── main.py                 # API routes + app factory
│   ├── models/schemas.py       # Pydantic request/response schemas
│   ├── services/
│   │   ├── food_classifier.py  # CNN inference (MobileNetV2)
│   │   ├── ingredient_service.py # Food → ingredient lookup
│   │   └── nutrition_service.py  # ANN nutrition prediction
│   └── utils/
│       ├── image_utils.py      # Image preprocessing
│       └── nutrition_utils.py  # Portion scaling + health score
├── streamlit_app/app.py        # Streamlit frontend
├── training/
│   ├── train_cnn.py            # CNN training script (Food-101)
│   ├── train_ann.py            # ANN training script
│   └── generate_mock_models.py # Quick-start without training
├── models/                     # Saved model files (generated)
├── data/
│   ├── food_ingredients.json   # 30 foods × ingredients (grams)
│   └── ingredient_nutrition.json # Per-gram nutrition values
├── requirements.txt
└── .env.example
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
cd food-ai-app
pip install -r requirements.txt
```

### 2. Generate mock model files (skip full training for demo)

```bash
python training/generate_mock_models.py
```

### 3. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Start the Streamlit frontend (new terminal)

```bash
streamlit run streamlit_app/app.py
```

Open: http://localhost:8501

---

## Training Real Models

### CNN (Food Classification)

1. Download Food11 dataset from Kaggle:
   ```bash
   kaggle datasets download -d trolukovich/food11-image-dataset -p data/
   ```

2. Extract it:
   ```powershell
   # Windows PowerShell
   Expand-Archive data\food11-image-dataset.zip -DestinationPath data\food11
   ```
   ```bash
   # Linux/Mac
   unzip data/food11-image-dataset.zip -d data/food11
   ```

   Expected structure after extraction:
   ```
   data/food11/
       training/   0/ 1/ 2/ ... 10/
       validation/ 0/ 1/ 2/ ... 10/
       evaluation/ 0/ 1/ 2/ ... 10/
   ```

3. Train:
   ```bash
   python training/train_cnn.py
   ```
   Saves `models/food_cnn.h5` and `models/class_labels.json`.

### ANN (Nutrition Prediction)

```bash
python training/train_ann.py
```
Saves `models/nutrition_ann.h5`, `models/scaler.pkl`, `models/feature_names.json`.

---

## API Reference

### `POST /predict`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Food image (JPEG/PNG) |
| `portion_size` | float (optional) | Desired portion in grams |

**Response:**
```json
{
  "food": "pizza",
  "confidence": 0.92,
  "ingredients": ["cheese", "tomato", "flour"],
  "nutrition": { "calories": 285, "protein": 12, "carbs": 36, "fat": 10 },
  "portion_size": 200,
  "adjusted_nutrition": { "calories": 570, "protein": 24, "carbs": 72, "fat": 20 },
  "health_score": "Moderate",
  "base_weight": 245
}
```

### `GET /foods` – List all supported food labels  
### `GET /health` – Service health check

---

## Deployment

### Backend → Render / Railway

1. Push to GitHub
2. Create a new Web Service pointing to your repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend → Streamlit Cloud

1. Push `streamlit_app/app.py` to GitHub
2. Connect repo at https://share.streamlit.io
3. Set `BACKEND_URL` secret to your deployed backend URL

---

## Health Score Logic

| Score | Criteria |
|-------|----------|
| Healthy | Calories < 300 kcal AND Protein > 15g |
| Moderate | Moderate macros |
| Indulgent | High calories or fat |

---

## Supported Foods (Food11 – 11 categories)

| # | Label | Examples |
|---|-------|---------|
| 0 | bread | toast, baguette, rolls |
| 1 | dairy_product | cheese, milk, yogurt |
| 2 | dessert | cake, ice cream, cookies |
| 3 | egg | omelette, fried egg, boiled egg |
| 4 | fried_food | fries, fried chicken, tempura |
| 5 | meat | steak, burger, chicken |
| 6 | noodles_pasta | ramen, spaghetti, pad thai |
| 7 | rice | fried rice, sushi, bibimbap |
| 8 | seafood | salmon, sushi, shrimp |
| 9 | soup | ramen broth, tomato soup |
| 10 | vegetable_fruit | salad, fruit bowl |
