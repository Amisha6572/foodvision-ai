"""
ANN Training Script – Nutrition Prediction
Trains a Keras MLP on ingredient vectors → (calories, protein, carbs, fat)
Saves model to models/nutrition_ann.h5 and scaler to models/scaler.pkl
"""

import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model

# ── Config ──────────────────────────────────────────────────────────────────
INGREDIENTS_FILE = "data/food_ingredients.json"
NUTRITION_FILE   = "data/ingredient_nutrition.json"
MODEL_OUT        = "models/nutrition_ann.h5"
SCALER_OUT       = "models/scaler.pkl"
FEATURES_OUT     = "models/feature_names.json"
# ────────────────────────────────────────────────────────────────────────────


def compute_nutrition(ingredients: dict, nutrition_db: dict) -> dict:
    """Compute total nutrition for a food item from its ingredient quantities."""
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    for ing, grams in ingredients.items():
        if ing in nutrition_db:
            n = nutrition_db[ing]
            totals["calories"] += n["calories"] * grams
            totals["protein"]  += n["protein"]  * grams
            totals["carbs"]    += n["carbs"]     * grams
            totals["fat"]      += n["fat"]       * grams
    return totals


def build_dataset(food_db: dict, nutrition_db: dict):
    """Build feature matrix X and target matrix y from food data."""
    # Collect all unique ingredients
    all_ingredients = sorted({
        ing
        for food in food_db.values()
        for ing in food["ingredients"]
    })

    X, y = [], []
    for food_name, food_data in food_db.items():
        ing_map = food_data["ingredients"]
        total_weight = sum(ing_map.values())

        # Feature vector: ingredient weight + ratio + total weight
        feature_vec = []
        for ing in all_ingredients:
            grams = ing_map.get(ing, 0.0)
            ratio = grams / total_weight if total_weight > 0 else 0.0
            feature_vec.extend([grams, ratio])
        feature_vec.append(total_weight)

        nutrition = compute_nutrition(ing_map, nutrition_db)
        X.append(feature_vec)
        y.append([
            nutrition["calories"],
            nutrition["protein"],
            nutrition["carbs"],
            nutrition["fat"]
        ])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), all_ingredients


def build_ann(input_dim: int) -> Model:
    """Build a small MLP regressor."""
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(4, activation="linear")(x)   # calories, protein, carbs, fat
    return Model(inputs=inp, outputs=out)


def train():
    os.makedirs("models", exist_ok=True)

    with open(INGREDIENTS_FILE) as f:
        food_db = json.load(f)
    with open(NUTRITION_FILE) as f:
        nutrition_db = json.load(f)

    X, y, feature_names = build_dataset(food_db, nutrition_db)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Save feature names for inference
    with open(FEATURES_OUT, "w") as f:
        json.dump(feature_names, f, indent=2)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_OUT}")

    # Augment dataset with small noise for robustness (since dataset is small)
    augmented_X, augmented_y = [X_scaled], [y]
    for _ in range(50):
        noise = np.random.normal(0, 0.05, X_scaled.shape)
        augmented_X.append(X_scaled + noise)
        augmented_y.append(y)
    X_aug = np.vstack(augmented_X)
    y_aug = np.vstack(augmented_y)

    X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.15, random_state=42)

    model = build_ann(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mean_absolute_error"]   # avoid 'mse' alias – not serializable in all Keras versions
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_mean_absolute_error"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=1, monitor="val_mean_absolute_error"),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_loss")
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nANN model saved to {MODEL_OUT}")


if __name__ == "__main__":
    train()
