import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

# Emotion labels (RAF-DB uses 1-indexed, we convert to 0-indexed)
EMOTION_LABELS = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}

NUM_CLASSES = len(EMOTION_LABELS)

# Model configuration
IMAGE_SIZE = 100  # RAF-DB aligned images are 100x100

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
