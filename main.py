"""
main.py — End-to-End Pipeline Runner
======================================
Microplastic Particle Detection and Shape-Based Classification
using Digital Image Processing and Machine Learning

This script runs the complete pipeline:
1. Load dataset and annotations
2. Process images through DIP pipeline
3. Extract geometric features from annotated particles
4. Assign pseudo-labels (Fiber / Fragment / Bead)
5. Save features to CSV
6. Train/test split
7. Train 3 classifiers (Random Forest, SVM, KNN)
8. Evaluate and compare models
9. Generate final labeled output images
10. Save all visualizations, reports, and models

Author: Student Project
Course: Digital Image Processing
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Suppress matplotlib and sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.utils import logger, ensure_output_dirs, safe_load_image
from src.data_loader import (
    load_annotations, load_image, get_image_list,
    display_sample_images, print_dataset_stats, get_image_annotations
)
from src.preprocessing import preprocess_pipeline, preprocess_particle_crop
from src.segmentation import segment_particles, find_contours, filter_contours
from src.feature_extraction import compute_features, NUMERIC_FEATURES
from src.labeling import label_all_particles, get_rule_based_predictions
from src.classifier import (
    prepare_data, train_models, compare_models,
    plot_feature_importance, run_ablation_study, save_model
)
from src.visualization import (
    plot_preprocessing_stages, plot_feature_distributions,
    plot_class_distribution, plot_pca, plot_tsne,
    draw_classified_image, plot_model_comparison
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# STEP 1: INITIALIZE
# ============================================================

def initialize():
    """Create output directories and print header."""
    print("\n" + "=" * 70)
    print("  MICROPLASTIC PARTICLE DETECTION & CLASSIFICATION")
    print("  Using Digital Image Processing + Machine Learning")
    print("=" * 70)
    
    ensure_output_dirs(config)
    print(f"\n  Project Root : {config.BASE_DIR}")
    print(f"  Dataset Dir  : {config.DATASET_DIR}")
    print(f"  Output Dir   : {config.OUTPUT_DIR}")


# ============================================================
# STEP 2: LOAD AND EXPLORE DATA
# ============================================================

def load_data():
    """Load annotations and display dataset statistics."""
    print("\n\n" + "-" * 50)
    print("STEP 1: Loading Dataset")
    print("-" * 50)
    
    train_df = load_annotations(config.TRAIN_ANNOTATIONS)
    valid_df = load_annotations(config.VALID_ANNOTATIONS)
    
    if train_df.empty:
        logger.error("Failed to load training annotations. Exiting.")
        sys.exit(1)
    
    print_dataset_stats(train_df, valid_df)
    
    # Save sample images grid
    display_sample_images(
        config.DATASET_DIR, train_df, n=config.NUM_SAMPLE_IMAGES,
        save_path=os.path.join(config.PLOTS_DIR, "sample_images.png")
    )
    
    return train_df, valid_df


# ============================================================
# STEP 3: DEMONSTRATE PREPROCESSING PIPELINE
# ============================================================

def demonstrate_preprocessing(train_df):
    """Show preprocessing stages on a sample image."""
    print("\n\n" + "-" * 50)
    print("STEP 2: Preprocessing Pipeline Demonstration")
    print("-" * 50)
    
    # Pick the first image that exists
    for fname in train_df["filename"].unique()[:10]:
        img = load_image(config.DATASET_DIR, fname)
        if img is not None:
            stages = preprocess_pipeline(img)
            save_path = os.path.join(
                config.PIPELINE_STAGES_DIR,
                f"pipeline_{os.path.splitext(fname)[0]}.png"
            )
            plot_preprocessing_stages(stages, save_path)
            print(f"  Pipeline demo saved for: {fname}")
            return stages
    
    logger.warning("Could not find any valid image for preprocessing demo")
    return None


# ============================================================
# STEP 4: EXTRACT FEATURES FROM ALL PARTICLES
# ============================================================

def extract_features_from_dataset(annotations_df, img_dir):
    """
    Process all images: crop particles using bounding boxes,
    preprocess, segment, and extract geometric features.
    
    Parameters:
        annotations_df: DataFrame with bounding box annotations
        img_dir: Directory containing images
    
    Returns:
        list: List of feature dictionaries for all particles
    """
    print("\n\n" + "-" * 50)
    print("STEP 3: Feature Extraction from All Particles")
    print("-" * 50)
    
    all_features = []
    unique_images = annotations_df["filename"].unique()
    total_images = len(unique_images)
    skipped = 0
    
    for img_idx, filename in enumerate(unique_images):
        # Progress indicator
        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            print(f"  Processing image {img_idx + 1}/{total_images}: {filename}")
        
        # Load image
        img = load_image(img_dir, filename)
        if img is None:
            skipped += 1
            continue
        
        # Get bounding boxes for this image
        img_annotations = get_image_annotations(annotations_df, filename)
        
        for _, row in img_annotations.iterrows():
            try:
                xmin = max(0, int(row["xmin"]))
                ymin = max(0, int(row["ymin"]))
                xmax = min(img.shape[1], int(row["xmax"]))
                ymax = min(img.shape[0], int(row["ymax"]))
                
                # Skip tiny bounding boxes
                if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                    continue
                
                # Crop the particle region using the bounding box
                crop = img[ymin:ymax, xmin:xmax]
                
                if crop.size == 0:
                    continue
                
                # Preprocess the cropped particle
                binary = preprocess_particle_crop(crop)
                
                # Find contours in the cropped binary image
                contours = find_contours(binary)
                contours = filter_contours(contours, min_area=20)
                
                if len(contours) == 0:
                    # No contours found — use the entire crop as one contour
                    # Create a rectangular contour from the bounding box
                    h_crop, w_crop = binary.shape[:2]
                    rect_contour = np.array([
                        [[0, 0]], [[w_crop-1, 0]], 
                        [[w_crop-1, h_crop-1]], [[0, h_crop-1]]
                    ], dtype=np.int32)
                    contours = [rect_contour]
                
                # Use the largest contour (main particle)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Extract features
                features = compute_features(largest_contour)
                
                if features is not None:
                    # Add image metadata
                    features["image_filename"] = filename
                    features["orig_xmin"] = xmin
                    features["orig_ymin"] = ymin
                    features["orig_xmax"] = xmax
                    features["orig_ymax"] = ymax
                    all_features.append(features)
            
            except Exception as e:
                logger.warning(f"  Error processing particle in {filename}: {e}")
                continue
    
    print(f"\n  Total particles processed: {len(all_features)}")
    print(f"  Images skipped/missing: {skipped}")
    
    return all_features


# ============================================================
# STEP 5: ASSIGN PSEUDO-LABELS
# ============================================================

def assign_labels(features_list):
    """Assign Fiber/Fragment/Bead labels using rule-based classification."""
    print("\n\n" + "-" * 50)
    print("STEP 4: Rule-Based Pseudo-Label Assignment")
    print("-" * 50)
    
    print(f"  Thresholds: {config.SHAPE_THRESHOLDS}")
    features_list = label_all_particles(features_list)
    
    return features_list


# ============================================================
# STEP 6: BUILD DATAFRAME AND SAVE CSV
# ============================================================

def build_dataframe(features_list):
    """Convert feature list to DataFrame and save to CSV."""
    print("\n\n" + "-" * 50)
    print("STEP 5: Building Feature DataFrame")
    print("-" * 50)
    
    df = pd.DataFrame(features_list)
    
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Feature Statistics:")
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    print(df[numeric_cols].describe().to_string())
    
    # Save CSV
    df.to_csv(config.FEATURES_CSV_PATH, index=False)
    print(f"\n  Saved features CSV: {config.FEATURES_CSV_PATH}")
    
    return df


# ============================================================
# STEP 7: TRAIN AND EVALUATE ML MODELS
# ============================================================

def train_and_evaluate(df):
    """Train models, evaluate, compare, and run ablation study."""
    print("\n\n" + "-" * 50)
    print("STEP 6: Machine Learning Training & Evaluation")
    print("-" * 50)
    
    # Prepare data with all features
    X, y, le, feature_names = prepare_data(df, NUMERIC_FEATURES)
    
    if X is None or len(X) < 20:
        logger.error("Not enough data for ML training!")
        return None, None, None, None
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"\n  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")
    
    # Train models
    print("\n  Training models...")
    models = train_models(X_train, y_train)
    
    # Compare all models
    comparison_df, best_model_name = compare_models(
        models, X_test, y_test, le,
        save_dir=config.PLOTS_DIR
    )
    
    # Feature importance from Random Forest
    rf_model = models["Random Forest"]["model"]
    plot_feature_importance(
        rf_model, feature_names,
        os.path.join(config.PLOTS_DIR, "feature_importance.png")
    )
    
    # Save best model
    save_model(models[best_model_name], best_model_name, config.MODELS_DIR)
    # Also save Random Forest (always useful for feature importance)
    save_model(models["Random Forest"], "Random Forest", config.MODELS_DIR)
    
    # Model comparison chart
    plot_model_comparison(
        comparison_df,
        os.path.join(config.PLOTS_DIR, "model_comparison.png")
    )
    
    # Run ablation study
    run_ablation_study(df, le, config.PLOTS_DIR)
    
    # Return for further use
    return models, le, best_model_name, feature_names


# ============================================================
# STEP 8: RULE-BASED vs ML COMPARISON
# ============================================================

def compare_rule_vs_ml(df, models, le, feature_names):
    """Compare rule-based classification accuracy against ML models."""
    print("\n\n" + "-" * 50)
    print("STEP 7: Rule-Based vs ML Comparison")
    print("-" * 50)
    
    # Rule-based predictions are already the pseudo-labels
    # ML needs test data
    X, y, _, _ = prepare_data(df, NUMERIC_FEATURES)
    if X is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Rule-based "accuracy" is 100% by definition (labels are from rules)
    # But we can compare how well ML generalizes
    print("\n  Note: Pseudo-labels ARE the rule-based predictions,")
    print("  so rule-based 'accuracy' on training data = 100%.")
    print("  The real comparison is how well ML generalizes on the test set:\n")
    
    for name, entry in models.items():
        model = entry["model"]
        scaler = entry["scaler"]
        X_input = scaler.transform(X_test) if scaler else X_test
        y_pred = model.predict(X_input)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        print(f"  {name:15s}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    print("\n  If ML accuracy is high, it means the geometric features")
    print("  are consistent and the rule-based labeling is reliable.\n")


# ============================================================
# STEP 9: ADD ML PREDICTIONS TO CSV
# ============================================================

def add_ml_predictions_to_df(df, models, best_model_name, feature_names):
    """Add ML prediction column to the DataFrame and re-save CSV."""
    print("\n\n" + "-" * 50)
    print("STEP 8: Adding ML Predictions to CSV")
    print("-" * 50)
    
    available = [c for c in feature_names if c in df.columns]
    df_valid = df.dropna(subset=available)
    
    best = models[best_model_name]
    X_all = df_valid[available].values
    
    if best["scaler"] is not None:
        X_all = best["scaler"].transform(X_all)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df_valid["pseudo_label"].values)
    
    y_pred_encoded = best["model"].predict(X_all)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    
    df.loc[df_valid.index, "ml_prediction"] = y_pred_labels
    
    # Re-save CSV with ML predictions
    df.to_csv(config.FEATURES_CSV_PATH, index=False)
    print(f"  Updated CSV with ML predictions: {config.FEATURES_CSV_PATH}")
    
    return df


# ============================================================
# STEP 10: GENERATE VISUALIZATIONS
# ============================================================

def generate_visualizations(df):
    """Generate all visualization plots."""
    print("\n\n" + "-" * 50)
    print("STEP 9: Generating Visualizations")
    print("-" * 50)
    
    # Feature distributions
    plot_feature_distributions(
        df, os.path.join(config.PLOTS_DIR, "feature_distributions.png")
    )
    
    # Class distribution
    plot_class_distribution(
        df, os.path.join(config.PLOTS_DIR, "class_distribution.png")
    )
    
    # PCA plot
    plot_pca(df, os.path.join(config.PLOTS_DIR, "pca_visualization.png"))
    
    # t-SNE plot
    plot_tsne(df, os.path.join(config.PLOTS_DIR, "tsne_visualization.png"))


# ============================================================
# STEP 11: GENERATE CLASSIFIED OUTPUT IMAGES
# ============================================================

def generate_classified_images(df, num_images=8):
    """
    Generate final output images with ML-predicted labels 
    drawn as colored bounding boxes.
    """
    print("\n\n" + "-" * 50)
    print("STEP 10: Generating Classified Output Images")
    print("-" * 50)
    
    unique_images = df["image_filename"].unique()
    sample_images = unique_images[:min(num_images, len(unique_images))]
    
    for fname in sample_images:
        img = load_image(config.DATASET_DIR, fname)
        if img is None:
            # Try valid dir
            img = load_image(config.VALID_DIR, fname)
        if img is None:
            continue
        
        # Get particles for this image
        img_particles = df[df["image_filename"] == fname].to_dict("records")
        
        # Draw classified image
        save_path = os.path.join(
            config.CLASSIFIED_IMAGES_DIR,
            f"classified_{os.path.splitext(fname)[0]}.png"
        )
        draw_classified_image(img, img_particles, save_path)
    
    print(f"  Generated {len(sample_images)} classified images")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Run the complete end-to-end pipeline."""
    start_time = time.time()
    
    # Step 1: Initialize
    initialize()
    
    # Step 2: Load data
    train_df, valid_df = load_data()
    
    # Step 3: Demonstrate preprocessing
    demonstrate_preprocessing(train_df)
    
    # Step 4: Extract features from ALL training particles
    print("\n  Processing TRAINING set...")
    train_features = extract_features_from_dataset(train_df, config.DATASET_DIR)
    
    # Also process validation set
    if not valid_df.empty:
        print("\n  Processing VALIDATION set...")
        valid_features = extract_features_from_dataset(valid_df, config.VALID_DIR)
        all_features = train_features + valid_features
    else:
        all_features = train_features
    
    if len(all_features) == 0:
        logger.error("No features extracted! Check dataset paths.")
        sys.exit(1)
    
    # Step 5: Assign pseudo-labels
    all_features = assign_labels(all_features)
    
    # Step 6: Build DataFrame and save CSV
    df = build_dataframe(all_features)
    
    # Step 7: Train and evaluate ML models
    result = train_and_evaluate(df)
    if result[0] is None:
        logger.error("ML training failed. Exiting.")
        sys.exit(1)
    models, le, best_model_name, feature_names = result
    
    # Step 8: Rule-based vs ML comparison
    compare_rule_vs_ml(df, models, le, feature_names)
    
    # Step 9: Add ML predictions to DataFrame
    df = add_ml_predictions_to_df(df, models, best_model_name, feature_names)
    
    # Step 10: Generate all visualizations
    generate_visualizations(df)
    
    # Step 11: Generate classified output images
    generate_classified_images(df, num_images=10)
    
    # Done!
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE!")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Best model  : {best_model_name}")
    print(f"  Features CSV: {config.FEATURES_CSV_PATH}")
    print(f"  Models saved: {config.MODELS_DIR}")
    print(f"  Plots saved : {config.PLOTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import cv2  # Import here to check availability early
    main()
