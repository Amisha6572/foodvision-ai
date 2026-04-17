# Model Design Document (MDD)
## FoodVision AI – Food Detection, Ingredient & Nutrition Estimator

---

## 1. Project Overview

| Field | Details |
|---|---|
| Project Name | FoodVision AI |
| Version | 1.0.0 |
| Type | End-to-End AI Web Application |
| Domain | Computer Vision + Nutrition Analysis |
| Stack | HuggingFace Transformers, TensorFlow/Keras, FastAPI, Streamlit |

### Objective
Given a food image, the system must:
1. Identify the food item with a confidence score
2. Return a realistic ingredient list
3. Estimate calories, protein, carbs, and fat
4. Scale nutrition to a user-defined portion size
5. Assign a health score

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│  • Image upload   • Portion size input               │
│  • Results display (food, ingredients, nutrition)    │
└─────────────────────┬────────────────────────────────┘
                      │ POST /predict
                      ▼
┌──────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│  /predict  /foods  /health                           │
└──────┬──────────────┬──────────────┬─────────────────┘
       ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────────┐
│  Food      │ │ Ingredient │ │  Nutrition     │
│ Classifier │ │  Service   │ │   Service      │
│            │ │            │ │                │
│ ViT model  │ │ JSON DB    │ │ ANN Model      │
│ Food-101   │ │ 101 foods  │ │ + direct calc  │
└────────────┘ └────────────┘ └────────────────┘
```

---

## 3. Model 1 – Food Classifier (ViT)

### 3.1 Model Details

| Field | Value |
|---|---|
| Model | `nateraw/food` |
| Architecture | Vision Transformer (ViT-Base/16) |
| Source | HuggingFace Hub |
| Pretrained On | Food-101 (101,000 images, 101 classes) |
| Input | RGB image (any size, auto-resized) |
| Output | Top-5 class probabilities |
| Inference | CPU / GPU via PyTorch |
| Download Size | ~330 MB (cached after first run) |

### 3.2 Food-101 Classes (101 total)
apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare, beet_salad,
beignets, bibimbap, bread_pudding, breakfast_burrito, bruschetta, caesar_salad,
cannoli, caprese_salad, carrot_cake, ceviche, cheesecake, cheese_plate,
chicken_curry, chicken_quesadilla, chicken_wings, chocolate_cake,
chocolate_mousse, churros, clam_chowder, club_sandwich, crab_cakes,
creme_brulee, croque_madame, cup_cakes, deviled_eggs, donuts, dumplings,
edamame, eggs_benedict, escargots, falafel, filet_mignon, fish_and_chips,
foie_gras, french_fries, french_onion_soup, french_toast, fried_calamari,
fried_rice, frozen_yogurt, garlic_bread, gnocchi, greek_salad,
grilled_cheese_sandwich, grilled_salmon, guacamole, gyoza, hamburger,
hot_and_sour_soup, hot_dog, huevos_rancheros, hummus, ice_cream, lasagna,
lobster_bisque, lobster_roll_sandwich, macaroni_and_cheese, macarons,
miso_soup, mussels, nachos, omelette, onion_rings, oysters, pad_thai,
paella, pancakes, panna_cotta, peking_duck, pho, pizza, pork_chop,
poutine, prime_rib, pulled_pork_sandwich, ramen, ravioli, red_velvet_cake,
risotto, samosa, sashimi, scallops, seaweed_salad, shrimp_and_grits,
spaghetti_bolognese, spaghetti_carbonara, spring_rolls, steak,
strawberry_shortcake, sushi, tacos, takoyaki, tiramisu, tuna_tartare, waffles

### 3.3 Prediction Strategy
```
Top-20 predictions from ViT
        ↓
Filter: keep only labels present in ingredient DB
        ↓
Top-3 food candidates selected
        ↓
Return highest-confidence food match
        ↓
Keyword fallback if no match found
```

### 3.4 Why ViT over CNN?
| Aspect | CNN (MobileNetV2) | ViT (nateraw/food) |
|---|---|---|
| Food classes | ~60 (ImageNet) | 101 (Food-101) |
| Sushi detection | ✗ | ✓ |
| Ramen detection | ✗ | ✓ |
| Training needed | Yes | No (pretrained) |
| Accuracy | ~72% top-1 | ~90% top-1 |

---

## 4. Model 2 – Nutrition ANN

### 4.1 Model Details

| Field | Value |
|---|---|
| Architecture | Keras MLP (Multi-Layer Perceptron) |
| Input | Ingredient weight vector + ratios + total weight |
| Output | [calories, protein, carbs, fat] |
| Layers | Dense(256) → BN → Dropout(0.3) → Dense(128) → BN → Dropout(0.2) → Dense(64) → Dense(4) |
| Loss | MSE |
| Optimizer | Adam (lr=1e-3) |
| Training Data | food_ingredients.json × ingredient_nutrition.json |
| Fallback | Direct calculation from per-gram nutrition DB |

### 4.2 Feature Engineering
```
For each ingredient in the food item:
  feature_i = [grams, grams/total_weight]

Final feature vector:
  [ing_1_grams, ing_1_ratio, ing_2_grams, ing_2_ratio, ..., total_weight]
```

### 4.3 Training Strategy
- Dataset augmented 50× with Gaussian noise (σ=0.05) for robustness
- StandardScaler applied to all inputs
- EarlyStopping (patience=20) on val_MAE
- ReduceLROnPlateau (factor=0.5, patience=10)
- Best weights restored automatically

### 4.4 Fallback Calculation
When ANN is unavailable, nutrition is calculated directly:
```
calories = Σ (ingredient_grams × cal_per_gram)
protein  = Σ (ingredient_grams × protein_per_gram)
carbs    = Σ (ingredient_grams × carbs_per_gram)
fat      = Σ (ingredient_grams × fat_per_gram)
```

---

## 5. Data Design

### 5.1 food_ingredients.json
```json
{
  "sushi": {
    "ingredients": {
      "rice": 100,
      "salmon": 60,
      "nori": 5,
      "soy_sauce": 10,
      "avocado": 30,
      "cucumber": 20
    }
  }
}
```
- 101 food entries
- Ingredient quantities in grams (realistic serving sizes)

### 5.2 ingredient_nutrition.json
```json
{
  "salmon": {
    "calories": 2.08,
    "protein": 0.20,
    "carbs": 0.0,
    "fat": 0.13
  }
}
```
- 80+ ingredients
- Values are per gram (multiply by grams to get total)

---

## 6. Portion Scaling

```
adjusted_nutrition = base_nutrition × (portion_size / base_weight)

Example:
  base_weight   = 225g  (sum of all ingredient grams)
  portion_size  = 300g  (user input)
  scale_factor  = 300 / 225 = 1.33

  calories: 285 × 1.33 = 379 kcal
  protein:  14  × 1.33 = 18.6g
```

---

## 7. Health Score

| Condition | Points |
|---|---|
| Calories < 300 kcal | +2 |
| Protein > 15g | +2 |
| Fat < 10g | +1 |
| Carbs < 40g | +1 |

| Total Score | Label |
|---|---|
| 5 – 6 | 🥗 Healthy |
| 3 – 4 | ⚖️ Moderate |
| 0 – 2 | 🍰 Indulgent |

---

## 8. API Contract

### POST /predict

**Request:**
```
Content-Type: multipart/form-data
file:         <image file>
portion_size: 200  (optional, float, grams)
```

**Response:**
```json
{
  "food":         "Sushi",
  "confidence":   0.94,
  "ingredients":  ["rice", "salmon", "nori", "avocado", "cucumber"],
  "nutrition": {
    "calories": 285, "protein": 14, "carbs": 38, "fat": 8
  },
  "portion_size": 200,
  "adjusted_nutrition": {
    "calories": 253, "protein": 12, "carbs": 34, "fat": 7
  },
  "health_score": "Healthy",
  "base_weight":  225
}
```

**Error Responses:**
| Code | Reason |
|---|---|
| 400 | Invalid image format or empty file |
| 422 | Food detected but no ingredient data found |

---

## 9. What Is NOT Needed

| Component | Status | Reason |
|---|---|---|
| Food11 dataset | ❌ Not needed | Replaced by pretrained ViT |
| food_cnn.h5 | ❌ Not needed | ViT handles classification |
| train_cnn.py | ❌ Not needed | No CNN training required |
| food11/ folder | ❌ Can delete | ~600MB saved |
| food11-image-dataset.zip | ❌ Can delete | ~600MB saved |

**Free up space:**
```bash
Remove-Item -Recurse -Force data\food11
Remove-Item data\food11-image-dataset.zip
Remove-Item models\food_cnn.h5
```

---

## 10. What IS Needed

| Component | Purpose |
|---|---|
| `nateraw/food` (HuggingFace cache) | Food classification |
| `nutrition_ann.h5` | Nutrition prediction |
| `scaler.pkl` | ANN input normalization |
| `feature_names.json` | ANN feature order |
| `food_ingredients.json` | Ingredient lookup |
| `ingredient_nutrition.json` | Per-gram nutrition values |

---

## 11. Deployment

| Service | Platform | Purpose |
|---|---|---|
| FastAPI backend | Render / Railway | ML inference + API |
| Streamlit frontend | Streamlit Cloud | User interface |

```
GitHub → Render (backend) → https://foodvision-api.onrender.com
GitHub → Streamlit Cloud  → https://foodvision.streamlit.app
                            BACKEND_URL = https://foodvision-api.onrender.com
```
