"""
classifier.py — Machine Learning Training, Evaluation & Ablation
==================================================================
Train and compare three classifiers:
1. Random Forest
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)

Also includes:
- Ablation study (minimal vs full features)
- Feature importance analysis (Random Forest)
- Model saving with joblib
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src import config
from src.feature_extraction import NUMERIC_FEATURES, MINIMAL_FEATURES
from src.utils import logger, ensure_dir


# ============================================================
# PREPARE DATA FOR TRAINING
# ============================================================

def prepare_data(df, feature_columns=None):
    """
    Prepare feature matrix X and label vector y from DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with features and 'pseudo_label'
        feature_columns (list): Which columns to use as features.
                                Default: all NUMERIC_FEATURES
    
    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Encoded label vector
        le (LabelEncoder): Fitted label encoder
        feature_names (list): Feature column names used
    """
    if feature_columns is None:
        feature_columns = NUMERIC_FEATURES
    
    # Filter to only columns that exist in the DataFrame
    available = [c for c in feature_columns if c in df.columns]
    if len(available) == 0:
        logger.error("No valid feature columns found in DataFrame!")
        return None, None, None, None
    
    # Drop rows with NaN values in feature columns
    df_clean = df.dropna(subset=available + ["pseudo_label"]).copy()
    
    X = df_clean[available].values
    
    # Encode labels: Fiber=0, Fragment=1, Bead=2 (or whatever order)
    le = LabelEncoder()
    y = le.fit_transform(df_clean["pseudo_label"].values)
    
    logger.info(f"  Prepared {len(X)} samples with {len(available)} features")
    logger.info(f"  Classes: {list(le.classes_)}")
    
    return X, y, le, available


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(X_train, y_train):
    """
    Train three classifiers: Random Forest, SVM, and KNN.
    
    Parameters:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
    
    Returns:
        dict: {model_name: fitted_model}
    """
    # Scale features for SVM and KNN (RF doesn't need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {}
    
    # 1. Random Forest — ensemble of decision trees
    #    Works well with mixed-scale features, provides feature importance
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    models["Random Forest"] = {"model": rf, "scaler": None}
    logger.info("  Trained Random Forest")
    
    # 2. SVM — finds optimal hyperplane separating classes
    #    Needs scaled features for best performance
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=config.RANDOM_STATE
    )
    svm.fit(X_train_scaled, y_train)
    models["SVM"] = {"model": svm, "scaler": scaler}
    logger.info("  Trained SVM")
    
    # 3. KNN — classifies based on k nearest neighbours
    #    Distance-based, so needs scaled features
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    )
    knn.fit(X_train_scaled, y_train)
    models["KNN"] = {"model": knn, "scaler": scaler}
    logger.info("  Trained KNN")
    
    return models


# ============================================================
# EVALUATE A SINGLE MODEL
# ============================================================

def evaluate_model(model_entry, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data.
    
    Parameters:
        model_entry (dict): {"model": fitted_model, "scaler": scaler_or_None}
        X_test (np.ndarray): Test features
        y_test (np.ndarray): True labels
        model_name (str): Name for display
    
    Returns:
        dict: Evaluation metrics
    """
    model = model_entry["model"]
    scaler = model_entry["scaler"]
    
    # Scale if needed
    if scaler is not None:
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test
    
    # Predict
    y_pred = model.predict(X_test_input)
    
    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    
    return metrics, y_pred


# ============================================================
# COMPARE ALL MODELS
# ============================================================

def compare_models(models, X_test, y_test, le, save_dir=None):
    """
    Evaluate and compare all trained models.
    
    Parameters:
        models (dict): Dictionary of model entries
        X_test (np.ndarray): Test features
        y_test (np.ndarray): True labels
        le (LabelEncoder): Label encoder for class names
        save_dir (str): Directory to save reports
    
    Returns:
        pd.DataFrame: Comparison table
        str: Name of the best model
    """
    results = []
    best_model_name = None
    best_f1 = -1
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    for name, entry in models.items():
        metrics, y_pred = evaluate_model(entry, X_test, y_test, name)
        results.append(metrics)
        
        # Track best model
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name
        
        # Print detailed classification report
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        
        # Save confusion matrix plot
        if save_dir:
            plot_confusion_matrix(
                y_test, y_pred, le.classes_, name,
                os.path.join(save_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
            )
    
    # Build comparison DataFrame
    comparison_df = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(comparison_df.to_string(index=False))
    print(f"\n** Best Model: {best_model_name} (F1={best_f1:.4f})")
    print("=" * 70 + "\n")
    
    # Save comparison table
    if save_dir:
        comparison_df.to_csv(
            os.path.join(save_dir, "model_comparison.csv"),
            index=False
        )
    
    return comparison_df, best_model_name


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Generate and save a confusion matrix heatmap.
    
    Parameters:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Where to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved confusion matrix: {save_path}")


# ============================================================
# FEATURE IMPORTANCE (Random Forest)
# ============================================================

def plot_feature_importance(rf_model, feature_names, save_path):
    """
    Plot feature importance from a trained Random Forest.
    
    This shows which geometric features are most useful for
    classifying particles — great for viva explanation.
    
    Parameters:
        rf_model: Fitted RandomForestClassifier
        feature_names (list): Names of features
        save_path (str): Where to save the plot
    """
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    
    ax.barh(
        range(len(feature_names)),
        importances[sorted_idx],
        color=colors
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Random Forest - Feature Importance Analysis", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved feature importance: {save_path}")
    
    # Print top 5 features
    print("\nTop 5 Most Important Features:")
    for rank, idx in enumerate(sorted_idx[:5]):
        print(f"  {rank+1}. {feature_names[idx]:20s} : {importances[idx]:.4f}")


# ============================================================
# ABLATION STUDY
# ============================================================

def run_ablation_study(df, le, save_dir):
    """
    Compare model performance with different feature sets:
    1. Minimal: Area + Perimeter only
    2. Full: All 14+ features
    
    Parameters:
        df (pd.DataFrame): Full feature DataFrame
        le (LabelEncoder): Label encoder
        save_dir (str): Directory to save results
    
    Returns:
        pd.DataFrame: Ablation results table
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY - Feature Set Comparison")
    print("=" * 70)
    
    results = []
    
    for label, feat_cols in [("Minimal (Area+Perimeter)", MINIMAL_FEATURES),
                              ("Full (All Features)", NUMERIC_FEATURES)]:
        available = [c for c in feat_cols if c in df.columns]
        if len(available) == 0:
            continue
        
        X, y, _, _ = prepare_data(df, available)
        if X is None:
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Train Random Forest for comparison
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=config.RANDOM_STATE
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        results.append({
            "Feature Set": label,
            "Num Features": len(available),
            "Accuracy": f"{acc:.4f}",
            "F1-Score": f"{f1:.4f}"
        })
        
        print(f"\n  {label}:")
        print(f"    Features: {available}")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1-Score: {f1:.4f}")
    
    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(os.path.join(save_dir, "ablation_results.csv"), index=False)
    print("\n" + "=" * 70 + "\n")
    
    return ablation_df


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model_entry, model_name, save_dir):
    """
    Save a trained model using joblib.
    
    Parameters:
        model_entry (dict): {"model": model, "scaler": scaler}
        model_name (str): Name of the model
        save_dir (str): Output directory
    """
    ensure_dir(save_dir)
    
    # Save model
    model_path = os.path.join(
        save_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl"
    )
    joblib.dump(model_entry["model"], model_path)
    
    # Save scaler if present
    if model_entry["scaler"] is not None:
        scaler_path = os.path.join(
            save_dir, f"{model_name.lower().replace(' ', '_')}_scaler.pkl"
        )
        joblib.dump(model_entry["scaler"], scaler_path)
    
    logger.info(f"  Saved model: {model_path}")
