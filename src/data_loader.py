"""
data_loader.py — Dataset Loading and Annotation Parsing
=========================================================
Functions to:
- Load the annotations CSV
- List all image files
- Load individual images safely
- Display sample images in a grid
"""

import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils import safe_load_image, logger
from src import config


# ============================================================
# LOAD ANNOTATIONS FROM CSV
# ============================================================

def load_annotations(csv_path):
    """
    Load the annotations CSV file containing bounding box data.
    
    The CSV has columns: filename, width, height, class, xmin, ymin, xmax, ymax
    
    Parameters:
        csv_path (str): Path to _annotations.csv
    
    Returns:
        pandas.DataFrame: Annotation data, or empty DataFrame on error
    """
    try:
        if not os.path.exists(csv_path):
            logger.error(f"Annotations file not found: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} annotations from {csv_path}")
        logger.info(f"  Unique images: {df['filename'].nunique()}")
        logger.info(f"  Classes found: {df['class'].unique().tolist()}")
        return df
    
    except Exception as e:
        logger.error(f"Error reading annotations: {e}")
        return pd.DataFrame()


# ============================================================
# GET LIST OF IMAGE FILES
# ============================================================

def get_image_list(folder):
    """
    List all .jpg and .png image files in a folder.
    
    Parameters:
        folder (str): Directory to scan
    
    Returns:
        list: Sorted list of image filenames (not full paths)
    """
    if not os.path.isdir(folder):
        logger.error(f"Directory not found: {folder}")
        return []
    
    valid_extensions = (".jpg", ".jpeg", ".png")
    images = [
        f for f in os.listdir(folder) 
        if f.lower().endswith(valid_extensions)
    ]
    images.sort()
    logger.info(f"Found {len(images)} images in {folder}")
    return images


# ============================================================
# LOAD A SINGLE IMAGE
# ============================================================

def load_image(img_dir, filename):
    """
    Load a single image by directory and filename.
    
    Parameters:
        img_dir (str): Directory containing the image
        filename (str): Image filename
    
    Returns:
        numpy.ndarray or None: Loaded image
    """
    path = os.path.join(img_dir, filename)
    return safe_load_image(path)


# ============================================================
# GET ANNOTATIONS FOR A SPECIFIC IMAGE
# ============================================================

def get_image_annotations(df, filename):
    """
    Get all bounding box annotations for a specific image.
    
    Parameters:
        df (pd.DataFrame): Full annotations DataFrame
        filename (str): Image filename to filter by
    
    Returns:
        pd.DataFrame: Rows for the given image
    """
    return df[df["filename"] == filename].reset_index(drop=True)


# ============================================================
# DISPLAY SAMPLE IMAGES
# ============================================================

def display_sample_images(img_dir, annotations_df, n=6, save_path=None):
    """
    Display a grid of sample images with bounding box annotations.
    
    Parameters:
        img_dir (str): Image directory
        annotations_df (pd.DataFrame): Annotations
        n (int): Number of sample images to show
        save_path (str): If provided, save the figure to this path
    """
    # Pick n unique image filenames
    unique_files = annotations_df["filename"].unique()
    sample_files = unique_files[:min(n, len(unique_files))]
    
    # Create subplot grid
    cols = 3
    rows = (len(sample_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()
    
    for idx, fname in enumerate(sample_files):
        img = load_image(img_dir, fname)
        if img is None:
            continue
        
        # Draw bounding boxes
        img_copy = img.copy()
        img_annots = get_image_annotations(annotations_df, fname)
        for _, row in img_annots.iterrows():
            cv2.rectangle(
                img_copy,
                (int(row["xmin"]), int(row["ymin"])),
                (int(row["xmax"]), int(row["ymax"])),
                (0, 255, 0), 2
            )
        
        # Convert BGR to RGB for matplotlib display
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{fname}\n({len(img_annots)} particles)", fontsize=8)
        axes[idx].axis("off")
    
    # Hide empty subplots
    for idx in range(len(sample_files), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Sample Microscope Images with Bounding Box Annotations", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        logger.info(f"Saved sample images plot: {save_path}")
    
    plt.close()


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_dataset_stats(train_df, valid_df=None):
    """
    Print summary statistics about the dataset.
    
    Parameters:
        train_df (pd.DataFrame): Training annotations
        valid_df (pd.DataFrame): Validation annotations (optional)
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\n--- Training Set ---")
    print(f"  Total annotations : {len(train_df)}")
    print(f"  Unique images     : {train_df['filename'].nunique()}")
    print(f"  Classes           : {train_df['class'].unique().tolist()}")
    
    # Bounding box size statistics
    train_df = train_df.copy()
    train_df["bb_width"] = train_df["xmax"] - train_df["xmin"]
    train_df["bb_height"] = train_df["ymax"] - train_df["ymin"]
    print(f"  Avg bbox width    : {train_df['bb_width'].mean():.1f} px")
    print(f"  Avg bbox height   : {train_df['bb_height'].mean():.1f} px")
    print(f"  Min bbox area     : {(train_df['bb_width'] * train_df['bb_height']).min():.0f} px²")
    print(f"  Max bbox area     : {(train_df['bb_width'] * train_df['bb_height']).max():.0f} px²")
    
    if valid_df is not None and len(valid_df) > 0:
        print(f"\n--- Validation Set ---")
        print(f"  Total annotations : {len(valid_df)}")
        print(f"  Unique images     : {valid_df['filename'].nunique()}")
    
    print("=" * 60 + "\n")
