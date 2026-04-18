"""
config.py — Central Configuration File
=======================================
All paths, thresholds, and constants are stored here.
Change values in this file to tune the entire pipeline.
"""

import os

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# Base project directory (parent of this file's folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, "_annotations.csv")
VALID_ANNOTATIONS = os.path.join(VALID_DIR, "_annotations.csv")

# Output directories (auto-created by main.py)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PIPELINE_STAGES_DIR = os.path.join(OUTPUT_DIR, "pipeline_stages")
CLASSIFIED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "classified_images")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FEATURES_CSV_PATH = os.path.join(OUTPUT_DIR, "features_full.csv")

# ============================================================
# 2. PREPROCESSING PARAMETERS
# ============================================================

# Gaussian Blur kernel size (must be odd)
GAUSSIAN_BLUR_KERNEL = (5, 5)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Adaptive Thresholding
ADAPTIVE_THRESH_BLOCK_SIZE = 15      # Neighbourhood size (odd number)
ADAPTIVE_THRESH_CONSTANT = 3         # Subtracted from mean

# Morphological Operations
MORPH_KERNEL_SIZE = (3, 3)           # Kernel for opening/closing
MORPH_OPEN_ITERATIONS = 1           # Iterations for opening
MORPH_CLOSE_ITERATIONS = 2          # Iterations for closing

# ============================================================
# 3. SEGMENTATION PARAMETERS
# ============================================================

# Minimum contour area to keep (filters noise)
MIN_CONTOUR_AREA = 100

# Whether to use Watershed segmentation for overlapping particles
USE_WATERSHED = True

# Minimum number of points to fit an ellipse (OpenCV requirement)
MIN_ELLIPSE_POINTS = 5

# ============================================================
# 4. SHAPE CLASSIFICATION THRESHOLDS (Rule-Based)
# ============================================================
# Tune these values to adjust Fiber / Bead / Fragment assignment.
# All thresholds are in one place for easy experimentation.

SHAPE_THRESHOLDS = {
    # Fiber: elongated, thin particles
    "fiber_aspect_ratio_min": 3.0,     # Width/Height must be > this
    "fiber_circularity_max": 0.3,      # Circularity must be < this

    # Bead: circular, nearly round particles
    "bead_circularity_min": 0.75,      # Circularity must be > this
    "bead_aspect_ratio_min": 0.8,      # Aspect ratio must be between
    "bead_aspect_ratio_max": 1.2,      # ... these two values
}

# ============================================================
# 5. MACHINE LEARNING PARAMETERS
# ============================================================

# Train/test split ratio
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Features used in "minimal" ablation experiment
MINIMAL_FEATURES = ["area", "perimeter"]

# Features used in "full" experiment (set dynamically in classifier.py)
# This is just a reference — the actual list is built from the DataFrame columns

# ============================================================
# 6. VISUALIZATION PARAMETERS
# ============================================================

# Colors for each particle class (BGR format for OpenCV)
CLASS_COLORS_BGR = {
    "Fiber": (0, 0, 255),       # Red
    "Fragment": (0, 165, 255),   # Orange
    "Bead": (0, 255, 0),        # Green
}

# Colors for Matplotlib (RGB normalized)
CLASS_COLORS_MPL = {
    "Fiber": "#e74c3c",
    "Fragment": "#f39c12",
    "Bead": "#2ecc71",
}

# Number of sample images to show in visualizations
NUM_SAMPLE_IMAGES = 6

# DPI for saved plots
PLOT_DPI = 150

# ============================================================
# 7. STREAMLIT SETTINGS
# ============================================================

STREAMLIT_TITLE = "Microplastic Particle Detector & Classifier"
MAX_UPLOAD_SIZE_MB = 10
