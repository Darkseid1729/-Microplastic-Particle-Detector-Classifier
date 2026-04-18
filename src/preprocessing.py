"""
preprocessing.py — Image Preprocessing Pipeline
=================================================
Applies the following DIP operations in sequence:
1. Convert to Grayscale
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Gaussian Blur for noise reduction
4. Adaptive Thresholding for binarization
5. Morphological Opening (remove small noise)
6. Morphological Closing (fill small holes)

Each function returns the processed image and can be used
independently or as part of the full pipeline.
"""

import cv2
import numpy as np
from src import config
from src.utils import logger


# ============================================================
# STEP 1: CONVERT TO GRAYSCALE
# ============================================================

def to_grayscale(image):
    """
    Convert a BGR color image to grayscale.
    
    Grayscale conversion reduces the image from 3 channels to 1,
    making it easier to apply thresholding and morphological operations.
    
    Parameters:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        numpy.ndarray: Grayscale image
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# STEP 2: CLAHE (Contrast Enhancement)
# ============================================================

def apply_clahe(gray):
    """
    Apply CLAHE to improve contrast in microscope images.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization) is better
    than standard histogram equalization because it:
    - Works on small regions (tiles) instead of the whole image
    - Prevents over-amplification of noise
    - Handles uneven illumination common in microscopy
    
    Parameters:
        gray (numpy.ndarray): Grayscale image
    
    Returns:
        numpy.ndarray: Contrast-enhanced grayscale image
    """
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    enhanced = clahe.apply(gray)
    return enhanced


# ============================================================
# STEP 3: GAUSSIAN BLUR (Noise Reduction)
# ============================================================

def apply_gaussian_blur(gray, ksize=None):
    """
    Apply Gaussian Blur to reduce random noise.
    
    Gaussian blur smooths the image using a weighted average of
    neighboring pixels, which helps remove high-frequency noise
    without losing too much edge information.
    
    Parameters:
        gray (numpy.ndarray): Grayscale image
        ksize (tuple): Kernel size, e.g., (5, 5). Must be odd numbers.
    
    Returns:
        numpy.ndarray: Blurred image
    """
    if ksize is None:
        ksize = config.GAUSSIAN_BLUR_KERNEL
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    return blurred


# ============================================================
# STEP 4: ADAPTIVE THRESHOLDING (Binarization)
# ============================================================

def apply_adaptive_threshold(blurred):
    """
    Apply Adaptive Thresholding to convert grayscale to binary.
    
    Unlike global thresholding (Otsu's), adaptive thresholding
    calculates a different threshold for each small region of the image.
    This is crucial for microscope images where illumination is uneven.
    
    Method: ADAPTIVE_THRESH_GAUSSIAN_C
    - Uses weighted sum (Gaussian) of neighbourhood values
    - Better for images with gradual lighting changes
    
    Parameters:
        blurred (numpy.ndarray): Blurred grayscale image
    
    Returns:
        numpy.ndarray: Binary image (0 and 255)
    """
    binary = cv2.adaptiveThreshold(
        blurred,
        255,                                    # Maximum value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,        # Gaussian weighted mean
        cv2.THRESH_BINARY_INV,                  # Invert: particles = white
        config.ADAPTIVE_THRESH_BLOCK_SIZE,      # Neighbourhood size
        config.ADAPTIVE_THRESH_CONSTANT         # Constant subtracted from mean
    )
    return binary


# ============================================================
# STEP 5: MORPHOLOGICAL OPERATIONS
# ============================================================

def apply_morphology(binary):
    """
    Apply morphological opening and closing to clean the binary image.
    
    - Opening (erosion → dilation): Removes small noise spots
    - Closing (dilation → erosion): Fills small holes inside particles
    
    These operations use a rectangular kernel (structuring element).
    
    Parameters:
        binary (numpy.ndarray): Binary image
    
    Returns:
        numpy.ndarray: Cleaned binary image
    """
    # Create the structuring element (kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config.MORPH_KERNEL_SIZE
    )
    
    # Step 5a: Morphological Opening — remove small noise
    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel,
        iterations=config.MORPH_OPEN_ITERATIONS
    )
    
    # Step 5b: Morphological Closing — fill gaps in particles
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel,
        iterations=config.MORPH_CLOSE_ITERATIONS
    )
    
    return closed


# ============================================================
# FULL PREPROCESSING PIPELINE
# ============================================================

def preprocess_pipeline(image):
    """
    Run the complete preprocessing pipeline on a single image.
    
    Pipeline: Original → Grayscale → CLAHE → Blur → Threshold → Morphology
    
    Parameters:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        dict: Dictionary containing each intermediate result:
            - 'original': Original BGR image
            - 'grayscale': Grayscale image
            - 'clahe': CLAHE enhanced image
            - 'blurred': Gaussian blurred image
            - 'threshold': Adaptive thresholded binary image
            - 'morphology': Final cleaned binary image
    """
    stages = {}
    
    # Store original
    stages["original"] = image.copy()
    
    # Step 1: Grayscale
    gray = to_grayscale(image)
    stages["grayscale"] = gray
    
    # Step 2: CLAHE contrast enhancement
    enhanced = apply_clahe(gray)
    stages["clahe"] = enhanced
    
    # Step 3: Gaussian blur
    blurred = apply_gaussian_blur(enhanced)
    stages["blurred"] = blurred
    
    # Step 4: Adaptive thresholding
    binary = apply_adaptive_threshold(blurred)
    stages["threshold"] = binary
    
    # Step 5: Morphological operations
    cleaned = apply_morphology(binary)
    stages["morphology"] = cleaned
    
    return stages


# ============================================================
# PREPROCESS A CROPPED PARTICLE (from bounding box)
# ============================================================

def preprocess_particle_crop(crop):
    """
    Preprocess a small cropped particle image for contour extraction.
    
    This is a simplified pipeline optimized for already-cropped
    bounding box regions where we just need a clean binary mask.
    
    Parameters:
        crop (numpy.ndarray): Cropped BGR particle image
    
    Returns:
        numpy.ndarray: Binary mask of the particle
    """
    gray = to_grayscale(crop)
    enhanced = apply_clahe(gray)
    blurred = apply_gaussian_blur(enhanced, ksize=(3, 3))
    
    # Use Otsu's thresholding for small crops (more reliable than adaptive)
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Light morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary
