import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_cleaning.main import get_clean_data


def add_sidebar():
    st.sidebar.title("🧾 Enter Report Values")
    df = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"), ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"), ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"), ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"), ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"), ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"), ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"), ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"), ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"), ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"), ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(df[key].max()),
            value=float(df[key].mean())
        )

    submitted = st.sidebar.button("🔍 Get Report")
    return input_dict, submitted


def add_predictions(input_data, trigger):
    if not trigger:
        return

    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_arr = np.array(list(input_data.values())).reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)
    prediction = model.predict(input_arr_scaled)

    st.subheader("🧬 Prediction Result")
    if prediction[0] == 1:
        st.error("🔴 The model predicts the tumor is **Malignant**.", icon="🚨")
    else:
        st.success("🟢 The model predicts the tumor is **Benign**.", icon="✅")

    with st.expander("📊 Input Summary"):
        st.dataframe(pd.DataFrame(input_data, index=["Value"]).T)


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    input_data, submit = add_sidebar()

    with st.container():
        st.title("🩺 Breast Cancer Report")


    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.container():
        with st.expander("📝 See your Report", expanded=False):
            st.write('''
                To see your report:
                1. Click ">" at the top left.
                2. Enter the values from your diagnostic report.
                3. Click **Get Report**.
            ''')

        with st.expander("ℹ️ About this Predictor", expanded=False):
            st.markdown("""
            **Objective:**  
            To build an accurate, explainable model that assists medical professionals in detecting breast cancer.

            **How It Works:**  
            - Uses Support Vector Classifier (SVC)
            - Preprocessing: Standard Scaler  
            - Accuracy: **96.5%**

            **Dataset:**  
            - Extracted features from breast mass images  
            - Diagnosis: Malignant (M) or Benign (B)

            **Planned Features:**  
            - SHAP visualizations for model interpretability
            """)

    with st.container():
        add_predictions(input_data, submit)


if __name__ == "__main__":
    main()