"""
feature_extraction.py — Geometric Feature Extraction
======================================================
For every detected particle contour, extract 14+ features:

Basic Features:
1.  Area — number of pixels inside the contour
2.  Perimeter — total length of the contour boundary
3.  Width — bounding box width
4.  Height — bounding box height
5.  Aspect Ratio — Width / Height
6.  Circularity — 4πA / P² (1.0 for perfect circle)
7.  Solidity — Area / Convex Hull Area
8.  Extent — Area / Bounding Box Area
9.  Equivalent Diameter — √(4A / π)

Extended Features:
10. Convex Hull Area
11. Eccentricity — from fitted ellipse (0=circle, 1=line)
12. Major Axis Length — longest diameter of fitted ellipse
13. Minor Axis Length — shortest diameter of fitted ellipse
14-20. Hu Moments — 7 rotation-invariant shape descriptors
"""

import cv2
import numpy as np
import math
from src import config
from src.utils import safe_divide, logger


# ============================================================
# COMPUTE ALL FEATURES FOR ONE CONTOUR
# ============================================================

def compute_features(contour):
    """
    Extract all geometric features from a single contour.
    
    Parameters:
        contour (numpy.ndarray): A single OpenCV contour
    
    Returns:
        dict: Feature dictionary with all computed values,
              or None if the contour is invalid
    """
    try:
        features = {}
        
        # --------------------------------------------------
        # BASIC MEASUREMENTS
        # --------------------------------------------------
        
        # Area: number of pixels enclosed by the contour
        area = cv2.contourArea(contour)
        if area < 1:
            return None  # Skip degenerate contours
        features["area"] = area
        
        # Perimeter: total boundary length
        perimeter = cv2.arcLength(contour, closed=True)
        features["perimeter"] = perimeter
        
        # Bounding box: smallest upright rectangle containing the contour
        x, y, w, h = cv2.boundingRect(contour)
        features["bbox_x"] = x
        features["bbox_y"] = y
        features["width"] = w
        features["height"] = h
        
        # --------------------------------------------------
        # DERIVED SHAPE FEATURES
        # --------------------------------------------------
        
        # Aspect Ratio = Width / Height
        # Value > 1 means wider, < 1 means taller
        # Fibers typically have very high aspect ratio
        features["aspect_ratio"] = safe_divide(w, h, default=1.0)
        
        # Circularity = 4πA / P²
        # Perfect circle = 1.0, irregular shapes < 1.0
        # Beads should be close to 1.0, fibers close to 0
        features["circularity"] = safe_divide(
            4 * math.pi * area,
            perimeter * perimeter,
            default=0.0
        )
        
        # Convex Hull = tightest convex polygon around the contour
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features["convex_hull_area"] = hull_area
        
        # Solidity = Area / Convex Hull Area
        # Measures how "filled" the shape is. Irregular shapes have lower solidity.
        features["solidity"] = safe_divide(area, hull_area, default=0.0)
        
        # Extent = Area / Bounding Box Area
        # How much of the bounding box is filled by the contour
        bbox_area = w * h
        features["extent"] = safe_divide(area, bbox_area, default=0.0)
        
        # Equivalent Diameter = √(4A / π)
        # Diameter of a circle with the same area
        features["equiv_diameter"] = math.sqrt(safe_divide(4 * area, math.pi, default=0.0))
        
        # --------------------------------------------------
        # ELLIPSE FEATURES (Eccentricity, Major/Minor Axis)
        # --------------------------------------------------
        
        if len(contour) >= config.MIN_ELLIPSE_POINTS:
            try:
                # Fit an ellipse to the contour
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse
                
                features["major_axis"] = max(major_axis, minor_axis)
                features["minor_axis"] = min(major_axis, minor_axis)
                
                # Eccentricity: 0 = perfect circle, close to 1 = very elongated
                # e = √(1 - (b²/a²)) where a = semi-major, b = semi-minor
                a = features["major_axis"] / 2.0
                b = features["minor_axis"] / 2.0
                if a > 0:
                    eccentricity_sq = 1.0 - safe_divide(b * b, a * a, default=1.0)
                    features["eccentricity"] = math.sqrt(max(0, eccentricity_sq))
                else:
                    features["eccentricity"] = 0.0
                    
            except cv2.error:
                # Ellipse fitting can fail for very small or collinear contours
                features["major_axis"] = max(w, h)
                features["minor_axis"] = min(w, h)
                features["eccentricity"] = 0.0
        else:
            # Not enough points to fit ellipse — use bounding box as fallback
            features["major_axis"] = max(w, h)
            features["minor_axis"] = min(w, h)
            features["eccentricity"] = 0.0
        
        # --------------------------------------------------
        # HU MOMENTS (7 rotation-invariant descriptors)
        # --------------------------------------------------
        
        # Hu Moments are derived from central moments and are invariant
        # to translation, rotation, and scale. They are widely used in
        # shape recognition and pattern matching.
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Apply log transform to Hu Moments for better numerical range
        # log(|hu|) makes the values more comparable across shapes
        for i in range(7):
            value = hu_moments[i]
            if value != 0:
                features[f"hu_{i+1}"] = -np.sign(value) * np.log10(abs(value))
            else:
                features[f"hu_{i+1}"] = 0.0
        
        return features
    
    except Exception as e:
        logger.warning(f"Feature extraction failed for contour: {e}")
        return None


# ============================================================
# EXTRACT FEATURES FROM ALL CONTOURS
# ============================================================

def extract_all_features(contours, image_filename="", bbox_info=None):
    """
    Extract features from a list of contours.
    
    Parameters:
        contours (list): List of OpenCV contours
        image_filename (str): Source image filename (for the CSV)
        bbox_info (dict): Optional bounding box info {xmin, ymin, xmax, ymax}
    
    Returns:
        list: List of feature dictionaries (one per valid contour)
    """
    all_features = []
    
    for idx, contour in enumerate(contours):
        features = compute_features(contour)
        if features is not None:
            # Add metadata
            features["image_filename"] = image_filename
            features["particle_id"] = idx
            
            # Add original bounding box info if provided
            if bbox_info is not None:
                features["orig_xmin"] = bbox_info.get("xmin", 0)
                features["orig_ymin"] = bbox_info.get("ymin", 0)
                features["orig_xmax"] = bbox_info.get("xmax", 0)
                features["orig_ymax"] = bbox_info.get("ymax", 0)
            
            all_features.append(features)
    
    return all_features


# ============================================================
# FEATURE COLUMNS (for DataFrame operations)
# ============================================================

# List of numeric features used for ML training
NUMERIC_FEATURES = [
    "area", "perimeter", "width", "height",
    "aspect_ratio", "circularity", "solidity", "extent",
    "equiv_diameter", "convex_hull_area", "eccentricity",
    "major_axis", "minor_axis",
    "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7"
]

# Minimal feature set for ablation study
MINIMAL_FEATURES = ["area", "perimeter"]
