"""
labeling.py — Rule-Based Shape Classification
================================================
Assign pseudo-labels (Fiber, Fragment, Bead) to particles
based on geometric features using configurable thresholds.

Rule Logic:
-----------
1. FIBER:  aspect_ratio > threshold AND circularity < threshold
           → Long, thin, thread-like particles
           
2. BEAD:   circularity > threshold AND aspect_ratio in [min, max]
           → Circular, nearly round particles
           
3. FRAGMENT: Everything else
           → Irregular broken pieces

All thresholds are stored in config.SHAPE_THRESHOLDS for easy tuning.
"""

from src import config
from src.utils import logger


# ============================================================
# CLASSIFY A SINGLE PARTICLE
# ============================================================

def assign_shape_label(features, thresholds=None):
    """
    Assign a shape label to a particle based on its geometric features.
    
    The classification uses simple threshold rules on aspect ratio
    and circularity — these are the two most discriminative features
    for distinguishing between fiber, bead, and fragment shapes.
    
    Parameters:
        features (dict): Feature dictionary from feature_extraction
        thresholds (dict): Override thresholds (default: config values)
    
    Returns:
        str: One of "Fiber", "Bead", or "Fragment"
    """
    if thresholds is None:
        thresholds = config.SHAPE_THRESHOLDS
    
    aspect_ratio = features.get("aspect_ratio", 1.0)
    circularity = features.get("circularity", 0.5)
    
    # Rule 1: FIBER detection
    # Fibers are elongated: high aspect ratio + low circularity
    if (aspect_ratio > thresholds["fiber_aspect_ratio_min"] and
            circularity < thresholds["fiber_circularity_max"]):
        return "Fiber"
    
    # Rule 2: BEAD detection
    # Beads are circular: high circularity + aspect ratio near 1.0
    if (circularity > thresholds["bead_circularity_min"] and
            thresholds["bead_aspect_ratio_min"] <= aspect_ratio <= thresholds["bead_aspect_ratio_max"]):
        return "Bead"
    
    # Rule 3: FRAGMENT — default for everything else
    # Fragments are irregular: don't fit fiber or bead criteria
    return "Fragment"


# ============================================================
# LABEL ALL PARTICLES
# ============================================================

def label_all_particles(features_list, thresholds=None):
    """
    Assign shape labels to all particles in a feature list.
    
    Each feature dictionary gets a new key "pseudo_label" with
    the assigned shape category.
    
    Parameters:
        features_list (list): List of feature dictionaries
        thresholds (dict): Override thresholds (optional)
    
    Returns:
        list: Same list with "pseudo_label" added to each dict
    """
    label_counts = {"Fiber": 0, "Bead": 0, "Fragment": 0}
    
    for features in features_list:
        label = assign_shape_label(features, thresholds)
        features["pseudo_label"] = label
        label_counts[label] += 1
    
    # Log the distribution
    total = sum(label_counts.values())
    logger.info(f"  Pseudo-label distribution:")
    for cls, count in label_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        logger.info(f"    {cls:10s}: {count:5d} ({pct:.1f}%)")
    
    return features_list


# ============================================================
# RULE-BASED ACCURACY ANALYSIS
# ============================================================

def get_rule_based_predictions(features_list, thresholds=None):
    """
    Get rule-based predictions for comparison with ML models.
    
    This function returns just the labels (no modification to features)
    for use in the ablation study comparing rule-based vs ML.
    
    Parameters:
        features_list (list): List of feature dictionaries
        thresholds (dict): Override thresholds
    
    Returns:
        list: List of predicted labels ("Fiber", "Bead", "Fragment")
    """
    return [assign_shape_label(f, thresholds) for f in features_list]
