"""
Generate lightweight mock models for development/demo purposes.
Run this if you don't have a trained CNN yet – the app will still work
using the direct nutrition calculation fallback.

Usage:
    python training/generate_mock_models.py
"""

import json
import os
import pickle
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("models", exist_ok=True)

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

# 1. Save class labels
labels = {str(i): food for i, food in enumerate(KNOWN_FOODS)}
with open("models/class_labels.json", "w") as f:
    json.dump(labels, f, indent=2)
print("✓ class_labels.json saved")

# 2. Save feature names (from ingredient DB)
with open("data/food_ingredients.json") as f:
    food_db = json.load(f)

all_ingredients = sorted({
    ing
    for food in food_db.values()
    for ing in food["ingredients"]
})
with open("models/feature_names.json", "w") as f:
    json.dump(all_ingredients, f, indent=2)
print(f"✓ feature_names.json saved ({len(all_ingredients)} ingredients)")

# 3. Save a mock scaler
from sklearn.preprocessing import StandardScaler

n_features = len(all_ingredients) * 2 + 1
scaler = StandardScaler()
# Fit on dummy data so it has valid mean/scale
dummy = np.random.rand(100, n_features)
scaler.fit(dummy)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✓ scaler.pkl saved (mock)")

print("\nMock model files generated. The app will use direct nutrition calculation.")
print("Run training/train_cnn.py and training/train_ann.py to train real models.")
