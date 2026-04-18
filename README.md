# 🔬 Microplastic Particle Detection and Shape-Based Classification

**Using Digital Image Processing and Machine Learning**

A complete end-to-end system that detects microplastic particles in microscope images, segments them, extracts geometric features, and classifies them into **Fiber**, **Fragment**, and **Bead** categories using classical image processing techniques and traditional machine learning classifiers.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [DIP Techniques Used](#dip-techniques-used)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Sample Outputs](#sample-outputs)
- [Time Complexity Analysis](#time-complexity-analysis)
- [Limitations](#limitations)
- [Why Classical DIP Over Deep Learning](#why-classical-dip-over-deep-learning)
- [Future Enhancements](#future-enhancements)
- [Viva Questions & Answers](#viva-questions--answers)

---

## Project Overview

Microplastic pollution is a major environmental concern. Identifying and classifying these particles under a microscope is time-consuming and subjective. This project automates the process using:

1. **Digital Image Processing** — to preprocess, enhance, and segment microscope images
2. **Geometric Feature Engineering** — to extract shape descriptors from each particle
3. **Rule-Based Labeling** — to generate shape-based pseudo-labels (Fiber/Fragment/Bead)
4. **Machine Learning** — to train classifiers that learn from the geometric features

### Key Technologies
- Python 3.8+
- OpenCV (image processing)
- NumPy, Pandas (data handling)
- Scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)
- Streamlit (web interface)

---

## Installation

```bash
# 1. Navigate to the project folder
cd project

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Run the Full Pipeline
```bash
python main.py
```
This will:
- Process all images in the dataset
- Extract features from ~5400 annotated particles
- Train Random Forest, SVM, and KNN classifiers
- Save models, features CSV, plots, and classified images

### Run the Streamlit Web App
```bash
streamlit run app.py
```
This opens a browser-based interface where you can upload images and see real-time classification.

---

## Pipeline Architecture

```
┌──────────────────────┐
│  1. Load Dataset &   │
│     Annotations CSV  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  2. Crop Particles   │
│     (Bounding Boxes) │
└──────────┬───────────┘
           ▼
┌──────────────────────────────────┐
│  3. DIP Preprocessing Pipeline   │
│  Grayscale → CLAHE → Blur →     │
│  Adaptive Threshold → Morphology │
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  4. Segmentation                 │
│  Contour Detection + Watershed   │
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  5. Feature Extraction (14+)     │
│  Area, Perimeter, Circularity,   │
│  Hu Moments, Eccentricity, etc.  │
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  6. Rule-Based Pseudo-Labeling   │
│  Fiber / Fragment / Bead         │
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  7. ML Training & Evaluation     │
│  Random Forest, SVM, KNN         │
│  + Ablation Study                │
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  8. Visualization & Outputs      │
│  Classified Images, Plots, CSV   │
└──────────────────────────────────┘
```

---

## Project Structure

```
project/
├── Dataset/                     # Microscope images + annotations
│   └── train/
│       ├── *.jpg                # ~577 training images
│       ├── _annotations.csv     # Bounding box annotations
│       └── valid/               # ~204 validation images
├── src/                         # Source modules
│   ├── config.py                # All paths, thresholds, constants
│   ├── data_loader.py           # Load images + CSV annotations
│   ├── preprocessing.py         # Grayscale, CLAHE, blur, threshold, morphology
│   ├── segmentation.py          # Contours + optional watershed
│   ├── feature_extraction.py    # 14+ geometric features + Hu moments
│   ├── labeling.py              # Configurable rule-based classification
│   ├── classifier.py            # Train/eval RF, SVM, KNN + ablation
│   ├── visualization.py         # All plots, PCA, t-SNE, confusion matrix
│   └── utils.py                 # Helpers, exception handling, logging
├── outputs/                     # Generated outputs (auto-created)
│   ├── pipeline_stages/         # Intermediate DIP images
│   ├── classified_images/       # Final labeled output images
│   ├── plots/                   # All visualization plots
│   ├── features_full.csv        # Complete feature + label CSV
│   └── models/                  # Saved ML models (.pkl)
├── main.py                      # End-to-end pipeline runner
├── app.py                       # Streamlit web frontend
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── report.md                    # Short academic report
```

---

## DIP Techniques Used

| Technique | Purpose | Why It's Needed |
|-----------|---------|-----------------|
| **Grayscale Conversion** | Reduce 3-channel to 1-channel | Simplifies processing |
| **CLAHE** | Adaptive contrast enhancement | Handles uneven microscope lighting |
| **Gaussian Blur** | Noise reduction | Removes high-frequency sensor noise |
| **Adaptive Thresholding** | Binarization | Separates particles from background |
| **Morphological Opening** | Remove small noise | Cleans up false detections |
| **Morphological Closing** | Fill gaps | Makes particle boundaries solid |
| **Contour Detection** | Find particle boundaries | Extracts individual particle shapes |
| **Watershed** | Separate overlapping particles | Handles touching/merged particles |

---

## Feature Extraction

### 14+ Feature for Each Particle

| Feature | Formula / Method | What It Measures |
|---------|-----------------|-----------------|
| Area | `cv2.contourArea()` | Size of the particle |
| Perimeter | `cv2.arcLength()` | Boundary length |
| Width, Height | `cv2.boundingRect()` | Bounding box dimensions |
| Aspect Ratio | `Width / Height` | Elongation |
| Circularity | `4πA / P²` | How circular (1.0 = perfect circle) |
| Solidity | `Area / ConvexHullArea` | How "filled" the shape is |
| Extent | `Area / BBoxArea` | BBox fill ratio |
| Equiv. Diameter | `√(4A/π)` | Circle-equivalent size |
| Eccentricity | From fitted ellipse | 0=circle, →1=elongated |
| Major/Minor Axis | `cv2.fitEllipse()` | Ellipse dimensions |
| Convex Hull Area | `cv2.convexHull()` | Tightest convex boundary |
| Hu Moments (×7) | `cv2.HuMoments()` | Rotation-invariant shape descriptors |

---

## Machine Learning Models

### Classifiers Trained
1. **Random Forest** — Ensemble of decision trees. Handles non-linear boundaries, provides feature importance.
2. **SVM** (Support Vector Machine) — Finds optimal separating hyperplane. Good for high-dimensional data.
3. **KNN** (K-Nearest Neighbors) — Distance-based classification. Simple and intuitive.

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (per class)
- Classification Report
- Feature Importance Analysis (Random Forest)

### Ablation Study
- **Minimal features** (Area + Perimeter only) vs **All features** (14+)
- Shows the value of geometric feature engineering

---

## Time Complexity Analysis

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Grayscale conversion | O(N×M) | Per-pixel operation |
| Gaussian Blur | O(N×M×k²) | k = kernel size |
| Adaptive Threshold | O(N×M×B²) | B = block size |
| Morphological Ops | O(N×M×k²×i) | i = iterations |
| Contour Detection | O(N×M) | Single pass |
| Feature Extraction | O(P) per contour | P = perimeter points |
| Watershed | O(N×M×log(N×M)) | Priority queue based |
| Random Forest Training | O(n×m×log(n)×T) | n=samples, m=features, T=trees |
| KNN Prediction | O(n×m) per query | n=training samples |

**Total per image**: O(N×M×B²) where N×M is image size, B is block size
**Total pipeline**: Scales linearly with number of images

---

## Limitations

### Pseudo-Labeling Limitations
1. **Rule-based labels are approximations** — they cannot capture all the nuances a human expert would notice
2. **Threshold sensitivity** — changing the aspect ratio or circularity thresholds even slightly can reclassify many particles
3. **Circular fragments** may be misclassified as beads
4. **Short fibers** with low aspect ratio may be classified as fragments

### Segmentation Challenges
1. **Overlapping particles** may merge into one contour (Watershed helps but isn't perfect)
2. **Very small particles** may be filtered as noise
3. **Background artifacts** (dust, scratches) can create false detections

### General Limitations
1. **No ground truth** — the dataset only has "Microplastic" labels, not shape-based truth labels
2. **Image quality varies** — some microscope images have severe noise or illumination issues
3. **Single-class annotation** — cannot validate pseudo-labels against expert-annotated shape labels

---

## Why Classical DIP Over Deep Learning

| Aspect | Classical DIP + ML | Deep Learning (Faster R-CNN) |
|--------|-------------------|-------------------------------|
| **Explainability** | ✅ Every step is interpretable | ❌ Black-box internal features |
| **Required Data** | Works with ~5K annotations | Needs 10K+ for good accuracy |
| **Compute** | CPU only, runs in seconds | Needs GPU, takes hours to train |
| **Feature Control** | ✅ You choose which features matter | ❌ Network learns unknown features |
| **Debugging** | ✅ Can inspect each intermediate step | ❌ Hard to diagnose failures |
| **Academic Value** | ✅ Demonstrates DIP fundamentals | ❌ Mostly framework usage |
| **Viva Friendly** | ✅ Can explain every formula | ❌ Hard to explain convolution layers |

---

## Future Enhancements

1. **Watershed Algorithm** improvements for better overlap handling
2. **GUI Application** using Tkinter or PyQt
3. **More particle categories** (film, pellet, foam)
4. **CNN-based classification** as a comparison experiment
5. **Real-time video** processing from microscope feed
6. **Active learning** — human-in-the-loop to correct pseudo-labels
7. **Multi-scale processing** for different magnification levels

---

## Viva Questions & Answers

### Q1: What is Adaptive Thresholding and why do you use it instead of global thresholding?
**A:** Adaptive thresholding calculates a different threshold for each small region of the image, using the weighted mean of the neighbourhood. This is crucial for microscope images because illumination is often uneven — one side may be brighter than the other. Global thresholding (like Otsu's) would use a single value for the entire image, which fails when lighting varies.

### Q2: Explain the difference between Morphological Opening and Closing.
**A:** Opening = Erosion followed by Dilation. It removes small bright spots (noise) while preserving larger objects. Closing = Dilation followed by Erosion. It fills small dark holes inside objects and connects nearby regions. We use Opening first to remove noise, then Closing to fill gaps in particle boundaries.

### Q3: What is Circularity and how is it calculated?
**A:** Circularity = 4πA / P², where A is the area and P is the perimeter. A perfect circle has circularity = 1.0. Irregular shapes have lower values. We use this to distinguish beads (high circularity ≈ 0.8-1.0) from fragments (medium ≈ 0.3-0.7) and fibers (low < 0.3).

### Q4: What is Aspect Ratio and why is it useful?
**A:** Aspect Ratio = Width / Height of the bounding box. Fibers are elongated, so they have high aspect ratio (>3). Beads are roughly square, so their aspect ratio is near 1.0. It's one of the most discriminative features for separating fibers from other shapes.

### Q5: What are Hu Moments?
**A:** Hu Moments are 7 values derived from central moments that are invariant to translation, rotation, and scale. They describe the shape of an object regardless of its position or orientation in the image. We use the log-transform of Hu Moments to make their numerical range more manageable.

### Q6: What is CLAHE and why is it better than regular histogram equalization?
**A:** CLAHE (Contrast Limited Adaptive Histogram Equalization) divides the image into small tiles and applies histogram equalization to each tile separately. It has a clip limit to prevent over-amplification of noise. Regular histogram equalization treats the whole image at once, which can wash out local details. CLAHE preserves local contrast, which is essential for detecting faint particles under uneven microscope lighting.

### Q7: Explain the Watershed algorithm.
**A:** Watershed treats the image as a topographic surface where pixel intensity represents elevation. It starts from known "marker" points (sure foreground) and "floods" outward. Where two flooding regions meet, a boundary (ridge) is created. We use it to separate overlapping particles that would otherwise merge into one contour.

### Q8: What is Solidity and what does it tell us about a particle?
**A:** Solidity = Contour Area / Convex Hull Area. The convex hull is the smallest convex polygon that contains all points of the contour. Solidity measures how "filled" the shape is relative to its convex hull. Fragments with irregular boundaries have lower solidity than beads.

### Q9: Why do you use Random Forest for feature importance?
**A:** Random Forest provides a built-in `feature_importances_` attribute that tells us how much each feature contributes to the classification decisions across all trees. This helps us understand which geometric properties are most discriminative and can be used to explain the model's behavior during a viva.

### Q10: What is the difference between rule-based and ML classification?
**A:** Rule-based classification uses manually defined thresholds (e.g., "if circularity > 0.75 → Bead"). It's simple, interpretable, but rigid. ML classification learns the decision boundaries from data. It can capture complex, non-linear patterns that simple rules miss. In our project, rule-based labels serve as "pseudo ground truth" to train the ML model, which can then generalize better.

### Q11: What is a Confusion Matrix?
**A:** A confusion matrix is an N×N table (where N is the number of classes) that shows how many samples were correctly or incorrectly classified. Rows represent the true class, columns represent the predicted class. Diagonal entries are correct predictions. Off-diagonal entries show misclassifications. It reveals which classes are most often confused with each other.

### Q12: Why did you use Adaptive Thresholding instead of Otsu's method?
**A:** Otsu's method finds a single optimal global threshold by minimizing intra-class variance. It works well when the histogram is bimodal (two clear peaks). But microscope images often have gradual lighting variations, making the histogram multimodal. Adaptive thresholding handles this by computing local thresholds, making it more robust for our use case.

### Q13: What is Eccentricity in the context of shape analysis?
**A:** Eccentricity is derived from fitting an ellipse to the contour. It's calculated as e = √(1 - b²/a²), where a is the semi-major axis and b is the semi-minor axis. A perfect circle has eccentricity = 0, a line has eccentricity close to 1. Fibers have high eccentricity, beads have low eccentricity.

### Q14: What is PCA and why do you use it for visualization?
**A:** PCA (Principal Component Analysis) is a dimensionality reduction technique that projects high-dimensional data onto new axes (principal components) that capture maximum variance. We use 2D PCA to visualize how well our 14+ features separate the three particle classes. If the classes form distinct clusters in PCA space, it confirms that our features are discriminative.

### Q15: How does the Gaussian Blur kernel size affect results?
**A:** Larger kernels (e.g., 7×7) produce more smoothing, removing more noise but also blurring edges. Smaller kernels (e.g., 3×3) preserve more detail but leave more noise. We use 5×5 as a balance. The kernel must be odd-sized so it has a center pixel.

### Q16: What is the Structuring Element in morphological operations?
**A:** The structuring element (or kernel) is a small matrix (e.g., 3×3) that defines the neighbourhood for erosion and dilation. Its shape (rectangular, elliptical, cross) affects which pixels are considered neighbours. We use rectangular kernels for general-purpose cleanup.

### Q17: How do you handle overlapping particles?
**A:** We use Watershed segmentation. First, we compute a distance transform to find the center of each particle. Then we threshold to find "sure foreground" markers. Watershed floods from these markers and creates boundaries where regions meet, effectively splitting merged particles.

### Q18: What is the Extent feature?
**A:** Extent = Contour Area / Bounding Box Area. It measures how much of the bounding rectangle is filled by the particle. Circular particles fill about π/4 ≈ 0.785 of their bounding box. Elongated fibers fill much less. It helps distinguish compact shapes from spread-out ones.

### Q19: Why do you normalize Hu Moments using log transform?
**A:** Hu Moments can span many orders of magnitude (from 10⁻¹ to 10⁻²⁰). Taking -sign(h)×log₁₀(|h|) compresses this range and makes the values more comparable. Without log transform, the larger moments would dominate any distance calculation.

### Q20: What would happen if you used only deep learning without DIP preprocessing?
**A:** Deep learning could potentially skip preprocessing and learn directly from raw pixels, but: (1) it needs much more data, (2) it's a black box — you can't explain the features, (3) it needs GPU hardware, (4) it's overkill for a structured problem with clear geometric rules. For a DIP course, showing command of classical techniques is more valuable and educational.
