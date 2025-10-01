# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide"
)

# --- Load Model and Data ---
model = joblib.load('iris_model.joblib')
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["Prediction", "Data Exploration"])

# --- Prediction Mode ---
if app_mode == "Prediction":
    st.title("🌸 Iris Species Prediction")
    st.markdown("This app predicts the species of an Iris flower based on its measurements.")

    st.sidebar.header("Input Features")
    st.sidebar.markdown("Use the sliders to input the flower's measurements.")

    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
        sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
        petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.3)
        petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
        data = {'sepal length (cm)': sepal_length,
                'sepal width (cm)': sepal_width,
                'petal length (cm)': petal_length,
                'petal width (cm)': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # --- Display User Input and Prediction ---
    st.subheader("Your Input:")
    st.write(input_df)

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction:")
    predicted_species = iris.target_names[prediction][0]
    
    if predicted_species == 'setosa':
        st.success(f"The predicted species is **{predicted_species.capitalize()}** 🌿")
    elif predicted_species == 'versicolor':
        st.info(f"The predicted species is **{predicted_species.capitalize()}** 🌷")
    else: # virginica
        st.warning(f"The predicted species is **{predicted_species.capitalize()}** 🌺")

    st.subheader('Prediction Probability:')
    proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.write(proba_df)


# --- Data Exploration Mode ---
elif app_mode == "Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("Explore the Iris dataset used to train the model.")
    
    st.sidebar.header("Visualization Options")
    
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Iris Dataset")
        st.write(iris_df)

    st.subheader("Feature Distribution")
    feature = st.selectbox("Select a feature for the histogram", iris.feature_names)
    fig_hist = px.histogram(iris_df, x=feature, color="species", title=f"Histogram of {feature}")
    st.plotly_chart(fig_hist)

    st.subheader("Feature Relationship")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis feature", iris.feature_names)
    with col2:
        y_axis = st.selectbox("Select Y-axis feature", iris.feature_names, index=1)
    
    fig_scatter = px.scatter(iris_df, x=x_axis, y=y_axis, color="species", title=f"{x_axis} vs. {y_axis}")
    st.plotly_chart(fig_scatter)