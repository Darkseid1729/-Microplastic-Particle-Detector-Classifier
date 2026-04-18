"""
visualization.py — Visualization and Plotting Functions
=========================================================
Generate and save:
- Preprocessing pipeline stage images
- Feature distribution histograms by class
- Class distribution bar chart
- PCA and t-SNE scatter plots
- Final classified images with labels and bounding boxes
- Model comparison bar chart
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_extraction import NUMERIC_FEATURES
from src.utils import logger, ensure_dir


# ============================================================
# PLOT PREPROCESSING PIPELINE STAGES
# ============================================================

def plot_preprocessing_stages(stages, save_path):
    """
    Display the preprocessing pipeline as a 2x3 grid of images.
    
    Shows: Original → Grayscale → CLAHE → Blurred → Threshold → Morphology
    
    Parameters:
        stages (dict): Dictionary from preprocess_pipeline()
        save_path (str): Path to save the figure
    """
    titles = ["Original", "Grayscale", "CLAHE Enhanced",
              "Gaussian Blur", "Adaptive Threshold", "Morphology"]
    keys = ["original", "grayscale", "clahe",
            "blurred", "threshold", "morphology"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (key, title) in enumerate(zip(keys, titles)):
        img = stages.get(key)
        if img is None:
            axes[idx].set_title(f"{title}\n(Not available)")
            axes[idx].axis("off")
            continue
        
        if len(img.shape) == 3:
            # Color image — convert BGR to RGB
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale or binary
            axes[idx].imshow(img, cmap="gray")
        
        axes[idx].set_title(title, fontsize=13, fontweight="bold")
        axes[idx].axis("off")
    
    plt.suptitle("Image Preprocessing Pipeline", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved pipeline stages: {save_path}")


# ============================================================
# FEATURE DISTRIBUTION HISTOGRAMS
# ============================================================

def plot_feature_distributions(df, save_path):
    """
    Plot histograms of key features, colored by particle class.
    
    Parameters:
        df (pd.DataFrame): Features DataFrame with 'pseudo_label'
        save_path (str): Where to save the figure
    """
    key_features = [
        "area", "perimeter", "aspect_ratio", "circularity",
        "solidity", "eccentricity"
    ]
    available = [f for f in key_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = config.CLASS_COLORS_MPL
    
    for idx, feat in enumerate(available):
        ax = axes[idx]
        for cls in ["Fiber", "Fragment", "Bead"]:
            data = df[df["pseudo_label"] == cls][feat].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.6, label=cls,
                       color=colors.get(cls, "#999999"))
        ax.set_title(feat.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
    
    # Hide empty subplots
    for idx in range(len(available), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Feature Distributions by Particle Class", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved feature distributions: {save_path}")


# ============================================================
# CLASS DISTRIBUTION BAR CHART
# ============================================================

def plot_class_distribution(df, save_path):
    """
    Bar chart showing count of Fiber, Fragment, Bead particles.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'pseudo_label' column
        save_path (str): Where to save
    """
    counts = df["pseudo_label"].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [config.CLASS_COLORS_MPL.get(c, "#999") for c in counts.index]
    
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add count labels on each bar
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha="center", fontsize=12, fontweight="bold")
    
    ax.set_xlabel("Particle Class", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Particle Class Distribution (Pseudo-Labels)", fontsize=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved class distribution: {save_path}")


# ============================================================
# PCA VISUALIZATION (2D)
# ============================================================

def plot_pca(df, save_path):
    """
    2D PCA scatter plot of particle features, colored by class.
    
    PCA (Principal Component Analysis) reduces the high-dimensional
    feature space to 2 dimensions for visualization. It preserves
    the directions of maximum variance.
    
    Parameters:
        df (pd.DataFrame): Features DataFrame with 'pseudo_label'
        save_path (str): Where to save
    """
    available = [f for f in NUMERIC_FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available + ["pseudo_label"]).copy()
    
    if len(df_clean) < 10:
        logger.warning("Not enough data for PCA plot")
        return
    
    X = df_clean[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls in ["Fiber", "Fragment", "Bead"]:
        mask = df_clean["pseudo_label"] == cls
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=cls, alpha=0.6, s=30,
            color=config.CLASS_COLORS_MPL.get(cls, "#999"),
            edgecolors="white", linewidths=0.3
        )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("PCA — 2D Projection of Particle Features", fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved PCA plot: {save_path}")


# ============================================================
# t-SNE VISUALIZATION (2D)
# ============================================================

def plot_tsne(df, save_path):
    """
    2D t-SNE scatter plot for nonlinear visualization.
    
    t-SNE is better than PCA at revealing cluster structure
    in high-dimensional data. It preserves local neighborhood
    relationships (nearby points stay nearby).
    
    Parameters:
        df (pd.DataFrame): Features DataFrame with 'pseudo_label'
        save_path (str): Where to save
    """
    available = [f for f in NUMERIC_FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available + ["pseudo_label"]).copy()
    
    if len(df_clean) < 30:
        logger.warning("Not enough data for t-SNE plot")
        return
    
    X = df_clean[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use perplexity proportional to dataset size (min 5, max 50)
    perplexity = min(50, max(5, len(df_clean) // 5))
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=config.RANDOM_STATE,
        max_iter=1000
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls in ["Fiber", "Fragment", "Bead"]:
        mask = df_clean["pseudo_label"] == cls
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            label=cls, alpha=0.6, s=30,
            color=config.CLASS_COLORS_MPL.get(cls, "#999"),
            edgecolors="white", linewidths=0.3
        )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE — Nonlinear Projection of Particle Features", fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved t-SNE plot: {save_path}")


# ============================================================
# DRAW FINAL CLASSIFIED IMAGE
# ============================================================

def draw_classified_image(image, particles, save_path=None):
    """
    Draw bounding boxes and predicted class labels on the original image.
    
    Parameters:
        image (numpy.ndarray): Original BGR image
        particles (list): List of dicts with keys:
            xmin, ymin, xmax, ymax, pseudo_label (and optionally ml_prediction)
        save_path (str): If provided, save the annotated image
    
    Returns:
        numpy.ndarray: Annotated image
    """
    result = image.copy()
    
    for p in particles:
        xmin = int(p.get("orig_xmin", p.get("bbox_x", 0)))
        ymin = int(p.get("orig_ymin", p.get("bbox_y", 0)))
        xmax = int(p.get("orig_xmax", p.get("bbox_x", 0) + p.get("width", 0)))
        ymax = int(p.get("orig_ymax", p.get("bbox_y", 0) + p.get("height", 0)))
        
        # Use ML prediction if available, otherwise pseudo-label
        label = p.get("ml_prediction", p.get("pseudo_label", "Unknown"))
        color = config.CLASS_COLORS_BGR.get(label, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            result,
            (xmin, ymin - label_size[1] - 6),
            (xmin + label_size[0] + 4, ymin),
            color, -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            result, label,
            (xmin + 2, ymin - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA
        )
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, result)
        logger.info(f"  Saved classified image: {save_path}")
    
    return result


# ============================================================
# MODEL COMPARISON BAR CHART
# ============================================================

def plot_model_comparison(comparison_df, save_path):
    """
    Bar chart comparing accuracy and F1-score of all models.
    
    Parameters:
        comparison_df (pd.DataFrame): From classifier.compare_models()
        save_path (str): Where to save
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    acc_vals = comparison_df["accuracy"].values
    f1_vals = comparison_df["f1_score"].values
    
    bars1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy", color="#3498db", edgecolor="black")
    bars2 = ax.bar(x + width/2, f1_vals, width, label="F1-Score", color="#e74c3c", edgecolor="black")
    
    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model_name"].values, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved model comparison chart: {save_path}")
