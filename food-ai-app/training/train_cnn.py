"""
CNN Training Script – Food Classification using MobileNetV2 + Transfer Learning
Trains on Food11 dataset (11 categories) and saves the model to models/food_cnn.h5

Food11 categories (folder name → label):
  0: Bread  1: Dairy product  2: Dessert   3: Egg        4: Fried food
  5: Meat   6: Noodles/Pasta  7: Rice      8: Seafood    9: Soup
  10: Vegetable/Fruit

Dataset structure expected:
  data/food11/training/   (or evaluation / validation)
  Each subfolder is a class (0..10).
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
DATA_DIR    = "data/food11"            # root folder of the Food11 dataset
TRAIN_DIR   = os.path.join(DATA_DIR, "training")
VAL_DIR     = os.path.join(DATA_DIR, "validation")
MODEL_OUT   = "models/food_cnn.h5"
LABELS_OUT  = "models/class_labels.json"

# Human-readable names for the 11 numeric folders
FOOD11_LABELS = {
    "0": "bread",
    "1": "dairy_product",
    "2": "dessert",
    "3": "egg",
    "4": "fried_food",
    "5": "meat",
    "6": "noodles_pasta",
    "7": "rice",
    "8": "seafood",
    "9": "soup",
    "10": "vegetable_fruit",
}
# ────────────────────────────────────────────────────────────────────────────


def build_model(num_classes: int) -> Model:
    """MobileNetV2 base + custom classification head."""
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze base layers initially
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


def get_generators():
    """
    Create train/val ImageDataGenerators.
    Food11 already has separate training/ and validation/ splits.
    """
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, val_data


def train():
    os.makedirs("models", exist_ok=True)

    # Check dataset exists
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}\n"
            "Make sure you extracted the Food11 zip into data/food11/"
        )

    print("Loading data generators...")
    train_data, val_data = get_generators()
    num_classes = len(train_data.class_indices)
    print(f"Found {num_classes} classes: {list(train_data.class_indices.keys())}")

    # Map numeric folder names → human-readable labels
    labels = {}
    for folder_name, idx in train_data.class_indices.items():
        human_label = FOOD11_LABELS.get(str(folder_name), folder_name)
        labels[str(idx)] = human_label

    with open(LABELS_OUT, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Class labels saved to {LABELS_OUT}")

    model, base = build_model(num_classes)

    # Phase 1 – train head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Phase 1 callbacks – aggressive early stopping since head trains fast
    callbacks_phase1 = [
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,                  # stop if no improvement for 4 epochs
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,                  # halve LR after 2 stagnant epochs
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print("\n=== Phase 1: Training classification head ===")
    model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks_phase1)

    # Phase 2 – fine-tune top layers of base
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),   # very low LR to avoid destroying pretrained weights
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Phase 2 callbacks – more patient since fine-tuning is slower
    callbacks_phase2 = [
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,                  # give fine-tuning more room to improve
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
    ]

    print("\n=== Phase 2: Fine-tuning top layers ===")
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks_phase2)

    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    train()
