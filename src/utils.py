"""
utils.py — Utility Functions and Exception Handling
=====================================================
Helper functions used across the project:
- Directory creation
- Safe image loading with error handling
- Logging setup
- Division safety
"""

import os
import cv2
import warnings
import logging

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger(name="microplastic", level=logging.INFO):
    """
    Create and return a logger with console output.
    
    Parameters:
        name (str): Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# Create a global logger for the project
logger = setup_logger()


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_dir(path):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
        path (str): Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def ensure_output_dirs(config):
    """
    Create all output directories defined in config.
    
    Parameters:
        config: The config module containing directory paths
    """
    dirs = [
        config.OUTPUT_DIR,
        config.PIPELINE_STAGES_DIR,
        config.CLASSIFIED_IMAGES_DIR,
        config.PLOTS_DIR,
        config.MODELS_DIR,
    ]
    for d in dirs:
        ensure_dir(d)
    logger.info(f"Output directories ready at: {config.OUTPUT_DIR}")


# ============================================================
# SAFE IMAGE LOADING
# ============================================================

def safe_load_image(image_path):
    """
    Safely load an image with error handling.
    
    Parameters:
        image_path (str): Full path to the image file
    
    Returns:
        numpy.ndarray or None: The loaded image, or None if loading fails
    """
    # Check if file exists
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return None
    
    # Try to read the image
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Corrupt or unreadable image: {image_path}")
            return None
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


# ============================================================
# SAFE MATH OPERATIONS
# ============================================================

def safe_divide(numerator, denominator, default=0.0):
    """
    Perform division with zero-division protection.
    
    This is critical for circularity calculation where 
    perimeter could be zero for degenerate contours.
    
    Parameters:
        numerator (float): The numerator
        denominator (float): The denominator
        default (float): Value to return if denominator is zero
    
    Returns:
        float: Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


# ============================================================
# CONTOUR VALIDATION
# ============================================================

def is_valid_contour(contour, min_area=100):
    """
    Check if a contour is valid for feature extraction.
    
    Parameters:
        contour: OpenCV contour
        min_area (int): Minimum area threshold
    
    Returns:
        bool: True if contour is valid
    """
    if contour is None or len(contour) == 0:
        return False
    
    area = cv2.contourArea(contour)
    if area < min_area:
        return False
    
    return True


# ============================================================
# IMAGE SAVING HELPER
# ============================================================

def save_image(image, path, name="image"):
    """
    Save an image to disk with error handling.
    
    Parameters:
        image: numpy.ndarray image to save
        path (str): Output file path
        name (str): Descriptive name for logging
    """
    try:
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, image)
        logger.info(f"Saved {name}: {path}")
    except Exception as e:
        logger.error(f"Failed to save {name} to {path}: {e}")
