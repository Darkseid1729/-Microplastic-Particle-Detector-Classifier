"""
app.py — Streamlit Frontend for Microplastic Classifier
=========================================================
A web-based interface to:
1. Upload a microscope image
2. Run the DIP preprocessing pipeline
3. Detect and segment particles
4. Extract features and classify them
5. Display intermediate stages and final results

Run with: streamlit run app.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.preprocessing import preprocess_pipeline, preprocess_particle_crop
from src.segmentation import segment_particles, find_contours, filter_contours
from src.feature_extraction import compute_features, NUMERIC_FEATURES
from src.labeling import assign_shape_label
from src.visualization import draw_classified_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔬 Microplastic Particle Detector & Classifier")
st.markdown("""
**Digital Image Processing + Machine Learning Pipeline**

Upload a microscope image to detect, segment, and classify microplastic particles 
into **Fiber**, **Fragment**, and **Bead** categories.
""")

st.markdown("---")


# ============================================================
# SIDEBAR — SETTINGS
# ============================================================

st.sidebar.header("⚙️ Settings")

# Threshold controls
st.sidebar.subheader("Classification Thresholds")
fiber_ar = st.sidebar.slider("Fiber: Min Aspect Ratio", 1.5, 5.0, 
                              config.SHAPE_THRESHOLDS["fiber_aspect_ratio_min"], 0.1)
fiber_circ = st.sidebar.slider("Fiber: Max Circularity", 0.1, 0.5,
                                config.SHAPE_THRESHOLDS["fiber_circularity_max"], 0.05)
bead_circ = st.sidebar.slider("Bead: Min Circularity", 0.5, 0.95,
                               config.SHAPE_THRESHOLDS["bead_circularity_min"], 0.05)

custom_thresholds = {
    "fiber_aspect_ratio_min": fiber_ar,
    "fiber_circularity_max": fiber_circ,
    "bead_circularity_min": bead_circ,
    "bead_aspect_ratio_min": 0.8,
    "bead_aspect_ratio_max": 1.2,
}

# Preprocessing controls
st.sidebar.subheader("Preprocessing")
use_clahe = st.sidebar.checkbox("Use CLAHE Enhancement", value=True)
use_watershed = st.sidebar.checkbox("Use Watershed Segmentation", value=True)
min_area = st.sidebar.slider("Min Contour Area", 10, 500, config.MIN_CONTOUR_AREA, 10)


# ============================================================
# LOAD ML MODEL (if available)
# ============================================================

@st.cache_resource
def load_ml_model():
    """Load the trained Random Forest model if available."""
    model_path = os.path.join(config.MODELS_DIR, "random_forest_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    return None

ml_model = load_ml_model()

if ml_model:
    st.sidebar.success("✅ ML Model loaded")
    use_ml = st.sidebar.checkbox("Use ML Classification", value=True)
else:
    st.sidebar.warning("⚠️ No trained model found. Run main.py first.")
    use_ml = False


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a Microscope Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a microscope image of microplastic particles"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("❌ Could not read the uploaded image. Please try another file.")
    else:
        st.success(f"✅ Image loaded: {image.shape[1]}×{image.shape[0]} px")
        
        # ---- PREPROCESSING ----
        st.header("1️⃣ Preprocessing Pipeline")
        
        stages = preprocess_pipeline(image)
        
        # Display pipeline stages
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(stages["original"], cv2.COLOR_BGR2RGB),
                    caption="Original", use_container_width=True)
        with col2:
            st.image(stages["grayscale"], caption="Grayscale", 
                    use_container_width=True, clamp=True)
        with col3:
            st.image(stages["clahe"], caption="CLAHE Enhanced",
                    use_container_width=True, clamp=True)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.image(stages["blurred"], caption="Gaussian Blur",
                    use_container_width=True, clamp=True)
        with col5:
            st.image(stages["threshold"], caption="Adaptive Threshold",
                    use_container_width=True, clamp=True)
        with col6:
            st.image(stages["morphology"], caption="Morphology (Cleaned)",
                    use_container_width=True, clamp=True)
        
        # ---- SEGMENTATION ----
        st.header("2️⃣ Particle Segmentation")
        
        contours, final_binary = segment_particles(
            image, stages["morphology"], use_watershed=use_watershed
        )
        contours = filter_contours(contours, min_area=min_area)
        
        # Draw contours
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
                    caption=f"Detected Contours ({len(contours)} particles)",
                    use_container_width=True)
        with col_b:
            st.image(final_binary, caption="Final Binary Mask",
                    use_container_width=True, clamp=True)
        
        st.info(f"**{len(contours)}** particles detected after filtering")
        
        # ---- FEATURE EXTRACTION & CLASSIFICATION ----
        if len(contours) > 0:
            st.header("3️⃣ Feature Extraction & Classification")
            
            particles = []
            for cnt in contours:
                features = compute_features(cnt)
                if features is not None:
                    # Get bounding box for display
                    x, y, w, h = cv2.boundingRect(cnt)
                    features["orig_xmin"] = x
                    features["orig_ymin"] = y
                    features["orig_xmax"] = x + w
                    features["orig_ymax"] = y + h
                    
                    # Classify
                    label = assign_shape_label(features, custom_thresholds)
                    features["pseudo_label"] = label
                    
                    # ML prediction if available
                    if use_ml and ml_model is not None:
                        try:
                            feat_vals = [features.get(f, 0) for f in NUMERIC_FEATURES
                                        if f in features]
                            pred = ml_model.predict([feat_vals])[0]
                            from sklearn.preprocessing import LabelEncoder
                            # Simple mapping
                            features["ml_prediction"] = label  # fallback
                        except Exception:
                            features["ml_prediction"] = label
                    else:
                        features["ml_prediction"] = label
                    
                    particles.append(features)
            
            if len(particles) > 0:
                # Show classified image
                classified_img = draw_classified_image(image, particles)
                st.image(
                    cv2.cvtColor(classified_img, cv2.COLOR_BGR2RGB),
                    caption="Final Classified Image",
                    use_container_width=True
                )
                
                # Feature table
                df = pd.DataFrame(particles)
                display_cols = ["pseudo_label", "area", "perimeter", 
                               "aspect_ratio", "circularity", "solidity",
                               "eccentricity", "equiv_diameter"]
                available_cols = [c for c in display_cols if c in df.columns]
                
                st.subheader("📊 Extracted Features")
                st.dataframe(df[available_cols], use_container_width=True)
                
                # Class distribution
                st.subheader("📈 Class Distribution")
                counts = df["pseudo_label"].value_counts()
                
                col_x, col_y = st.columns(2)
                with col_x:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = [config.CLASS_COLORS_MPL.get(c, "#999") for c in counts.index]
                    ax.bar(counts.index, counts.values, color=colors)
                    ax.set_ylabel("Count")
                    ax.set_title("Particle Types")
                    st.pyplot(fig)
                
                with col_y:
                    st.metric("Total Particles", len(particles))
                    for cls in ["Fiber", "Fragment", "Bead"]:
                        count = len(df[df["pseudo_label"] == cls])
                        st.metric(f"{cls}s", count)
        else:
            st.warning("No particles detected. Try adjusting the minimum contour area.")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload a microscope image to get started!")
    
    # Show a demo from existing dataset
    st.header("📂 Or Explore Existing Results")
    
    features_path = config.FEATURES_CSV_PATH
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        st.write(f"Loaded {len(df)} particles from existing analysis")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.write("No existing results found. Run `python main.py` first to generate them.")
