"""
segmentation.py — Particle Segmentation and Contour Detection
==============================================================
Functions to:
- Find contours in binary images
- Filter contours by area
- Apply Watershed segmentation for overlapping particles
- Extract individual particle regions
- Draw contours and bounding boxes on images
"""

import cv2
import numpy as np
from src import config
from src.utils import logger, is_valid_contour


# ============================================================
# FIND CONTOURS
# ============================================================

def find_contours(binary):
    """
    Find all contours in a binary image.
    
    Uses RETR_EXTERNAL to find only the outermost contours (ignores
    nested contours inside particles), and CHAIN_APPROX_SIMPLE to
    compress the contour representation.
    
    Parameters:
        binary (numpy.ndarray): Binary image (from thresholding)
    
    Returns:
        list: List of contours (each contour is a numpy array of points)
    """
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,          # Only outermost contours
        cv2.CHAIN_APPROX_SIMPLE     # Compress straight segments
    )
    return contours


# ============================================================
# FILTER CONTOURS BY AREA
# ============================================================

def filter_contours(contours, min_area=None):
    """
    Remove contours that are too small (noise) or too large (background).
    
    Parameters:
        contours (list): List of OpenCV contours
        min_area (int): Minimum area threshold (default from config)
    
    Returns:
        list: Filtered list of valid contours
    """
    if min_area is None:
        min_area = config.MIN_CONTOUR_AREA
    
    filtered = [c for c in contours if is_valid_contour(c, min_area)]
    return filtered


# ============================================================
# WATERSHED SEGMENTATION (for overlapping particles)
# ============================================================

def apply_watershed(image, binary):
    """
    Apply Watershed segmentation to separate overlapping particles.
    
    Watershed treats the image as a topographic surface where bright
    pixels are high peaks. It floods the surface from markers (sure
    foreground regions) and finds the ridges between basins, which
    become the segmentation boundaries.
    
    Steps:
    1. Distance Transform — find distance from each pixel to nearest zero
    2. Threshold the distance map to find "sure foreground" markers
    3. Determine "sure background" from dilation
    4. Unknown region = background - foreground
    5. Apply Watershed to label each region
    
    Parameters:
        image (numpy.ndarray): Original BGR image
        binary (numpy.ndarray): Binary mask (from morphology)
    
    Returns:
        numpy.ndarray: Updated binary mask with separated particles
    """
    try:
        # Step 1: Sure background — dilate the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        
        # Step 2: Distance transform — distance from each white pixel to nearest black
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Step 3: Sure foreground — threshold at 50% of max distance
        _, sure_fg = cv2.threshold(
            dist_transform, 
            0.5 * dist_transform.max(), 
            255, 
            cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)
        
        # Step 4: Unknown region = background - foreground
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Step 5: Label the markers
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1          # Background becomes 1 instead of 0
        markers[unknown == 255] = 0    # Unknown region becomes 0
        
        # Step 6: Apply Watershed
        img_color = image.copy()
        if len(img_color.shape) == 2:
            img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
        
        markers = cv2.watershed(img_color, markers)
        
        # Step 7: Create new binary mask from watershed result
        # Boundaries (markers == -1) are set to 0
        # Each labeled region is set to 255
        new_binary = np.zeros_like(binary)
        new_binary[markers > 1] = 255
        
        return new_binary
    
    except Exception as e:
        logger.warning(f"Watershed segmentation failed: {e}. Using original binary.")
        return binary


# ============================================================
# FULL SEGMENTATION PIPELINE
# ============================================================

def segment_particles(image, binary, use_watershed=None):
    """
    Complete segmentation pipeline: find and filter contours,
    optionally applying Watershed for overlapping particles.
    
    Parameters:
        image (numpy.ndarray): Original BGR image
        binary (numpy.ndarray): Binary mask (from preprocessing)
        use_watershed (bool): Override config.USE_WATERSHED
    
    Returns:
        list: List of filtered contours
        numpy.ndarray: Final binary mask (possibly refined by Watershed)
    """
    if use_watershed is None:
        use_watershed = config.USE_WATERSHED
    
    # Optionally apply watershed
    if use_watershed:
        binary = apply_watershed(image, binary)
    
    # Find contours
    contours = find_contours(binary)
    
    # Filter small contours (noise)
    contours = filter_contours(contours)
    
    logger.info(f"  Segmented {len(contours)} particles")
    
    return contours, binary


# ============================================================
# DRAW CONTOURS AND BOUNDING BOXES
# ============================================================

def draw_contours_on_image(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contour outlines on a copy of the image.
    
    Parameters:
        image (numpy.ndarray): Original image
        contours (list): List of contours
        color (tuple): BGR color for drawing
        thickness (int): Line thickness
    
    Returns:
        numpy.ndarray: Image with contours drawn
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def draw_bounding_boxes(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes around each contour.
    
    Parameters:
        image (numpy.ndarray): Original image
        contours (list): List of contours
        color (tuple): BGR color
        thickness (int): Line thickness
    
    Returns:
        numpy.ndarray: Image with bounding boxes drawn
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    return result
