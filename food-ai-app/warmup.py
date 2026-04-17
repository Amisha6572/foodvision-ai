"""
Run once during build to pre-download the nateraw/food model into cache.
This prevents timeout on first user request.

Usage (add to Render build command):
  pip install -r requirements.txt && python warmup.py
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Pre-downloading nateraw/food model...")
from transformers import pipeline
pipe = pipeline("image-classification", model="nateraw/food", top_k=3)
print("Model cached successfully.")
