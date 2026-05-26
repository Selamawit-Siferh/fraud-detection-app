import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans # Import KMeans for potentially deployed model

st.set_page_config(page_title="ML Assignment Deployment", layout="wide")
st.title("🚀 Anomaly Detection Deployment")

st.header("Suspicious Transaction Detection")
st.write("Enter transaction details below or upload a CSV file for anomaly detection.")

# Function to load model and preprocessor
@st.cache_resource
def load_anomaly_bundle():
    try:
        with open('anomaly_detection_model.pkl', 'rb') as f:
            bundle = pickle.load(f)
        return bundle['model'], bundle['preprocessor'], bundle['numerical_features'], bundle['categorical_features']
    except FileNotFoundError:
        st.error("Anomaly detection model bundle not found. Please train and save the model first.")
        return None, None, None, None

anomaly_model, preprocessor, numerical_features_saved, categorical_features_saved = load_anomaly_bundle()

if anomaly_model and preprocessor and numerical_features_saved is not None and categorical_features_saved is not None:
    # Determine the type of the loaded model for specific prediction logic if needed
    model_type = type(anomaly_model).__name__

    # Option 1: Individual Transaction Input
    st.subheader("Single Transaction Anomaly Prediction")

    # Create input fields for numerical features
    input_values_num = {}
    cols_num = st.columns(len(numerical_features_saved))
    for i, feature in enumerate(numerical_features_saved):
        with cols_num[i]:
            default_value = 0.0
            if feature == 'LocalAmount':
                default_value = 100.0
            elif 'hour' in feature:
                default_value = 12 # Midday
            input_values_num[feature] = st.number_input(f"{feature} (Numerical)", value=default_value, format="%.2f", key=f"single_{feature}_num")

    # Create input fields for categorical features
    input_values_cat = {}
    cols_cat = st.columns(len(categorical_features_saved))
    for i, feature in enumerate(categorical_features_saved):
        with cols_cat[i]:
            if feature == 'TransactionChannels':
                common_channels = ['Online', 'Branch', 'ATM', 'Mobile', 'Unknown'] # Example common channels, include 'Unknown'
                input_values_cat[feature] = st.selectbox(f"{feature} (Categorical)", options=common_channels, key=f"single_{feature}_cat")
            else:
                input_values_cat[feature] = st.text_input(f"{feature} (Categorical)", value=f"test_id_example", key=f"single_{feature}_cat")

    if st.button("Detect Anomaly in Single Transaction"): # Changed button text
        if anomaly_model and preprocessor:
            single_transaction_data = pd.DataFrame([{**input_values_num, **input_values_cat}])

            required_columns = numerical_features_saved + categorical_features_saved
            for col in required_columns:
                if col not in single_transaction_data.columns:
                    st.error(f"Missing input for required feature: {col}")
                    st.stop()

            scaled_single = preprocessor.transform(single_transaction_data[required_columns])

            # Adjust prediction logic based on model type
            if model_type == 'KMeans':
                # For KMeans, predict returns cluster labels. We need distance to centroid.
                distances = anomaly_model.transform(scaled_single)
                min_distance = np.min(distances, axis=1)[0]
                # Assuming a threshold was saved or can be re-calculated (for simplicity here, using a fixed one)
                # In a real scenario, you'd save the threshold or recompute it from training data distances.
                # For now, let's use a placeholder logic.
                # A better approach would be to save the threshold when saving the KMeans model.
                # For demonstration: if distance is unusually high, flag as anomaly.
                # This requires knowledge of the distribution of distances from training.
                # For deployment, a fixed threshold from training data's 99th percentile (or similar) should be used.
                # For now, let's use a very basic high value as an anomaly indicator if no threshold was explicitly saved
                # For example, if training max distance was 5, a distance of 10 might be an anomaly.
                # This is a simplification; a proper threshold must be determined during training.
                st.warning("K-Means deployed. Anomaly detection based on distance from centroids. A proper threshold needs to be determined from training data.")
                # As a placeholder, let's just show the distance.
                st.write(f"Distance to nearest centroid: {min_distance:.2f}")
                # You would typically have a saved `threshold` from training.
                # if min_distance > saved_kmeans_threshold: st.error(...) else: st.success(...)

                # For now, as a placeholder, let's assume if distance is very high, it's an anomaly (needs fine-tuning)
                if min_distance > 5: # Placeholder threshold, needs to be derived from training data
                     st.error("⚠️ Suspicious Transaction Detected! (K-Means distance based)")
                else:
                     st.success("✅ Normal Transaction. (K-Means distance based)")
            else:
                # For IsolationForest, OneClassSVM, DBSCAN (where -1 is anomaly)
                pred_single = anomaly_model.predict(scaled_single)
                if pred_single[0] == -1:
                    st.error("⚠️ Suspicious Transaction Detected!")
                else:
                    st.success("✅ Normal Transaction.")

    st.markdown("--- ")

    # Option 2: CSV File Upload
    st.subheader("Batch Anomaly Detection from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file for anomaly detection", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df_uploaded.head())

        if 'CreatedOn' in df_uploaded.columns and 'CreatedOn_hour' in numerical_features_saved:
            df_uploaded['CreatedOn_dt'] = pd.to_datetime(df_uploaded['CreatedOn'], errors='coerce')
            df_uploaded['CreatedOn_hour'] = df_uploaded['CreatedOn_dt'].dt.hour
        elif 'CreatedOn_hour' in numerical_features_saved and 'CreatedOn_hour' not in df_uploaded.columns:
            st.warning("Column 'CreatedOn' not found in uploaded CSV, and 'CreatedOn_hour' is a required numerical feature. Filling 'CreatedOn_hour' with 0 (default).")
            df_uploaded['CreatedOn_hour'] = 0

        df_uploaded.dropna(subset=[f for f in numerical_features_saved if f in df_uploaded.columns], inplace=True)

        for col in categorical_features_saved:
            if col not in df_uploaded.columns:
                st.warning(f"Column '{col}' not found in uploaded CSV. Adding column and filling with 'Unknown'.")
                df_uploaded[col] = 'Unknown'
            else:
                df_uploaded[col] = df_uploaded[col].fillna('Unknown')

        all_required_cols = numerical_features_saved + categorical_features_saved
        missing_from_uploaded = [col for col in all_required_cols if col not in df_uploaded.columns]
        if missing_from_uploaded:
            st.error(f"Error: Uploaded CSV and its processed form must contain all required features: {', '.join(all_required_cols)}. Missing: {', '.join(missing_from_uploaded)}")
            st.stop()

        X_uploaded = df_uploaded[all_required_cols]

        scaled_uploaded = preprocessor.transform(X_uploaded)

        # Adjust prediction logic based on model type for batch prediction
        if model_type == 'KMeans':
            distances_batch = anomaly_model.transform(scaled_uploaded)
            min_distances_batch = np.min(distances_batch, axis=1)
            # Apply the same threshold logic as in single prediction
            # Placeholder threshold, needs to be derived from training data
            threshold_kmeans = 5 # This must be consistent with training
            predictions_batch = np.where(min_distances_batch > threshold_kmeans, -1, 1)
        else:
            predictions_batch = anomaly_model.predict(scaled_uploaded)

        df_uploaded['Anomaly_Prediction'] = np.where(predictions_batch == -1, "Suspicious", "Normal")

        st.write("Anomaly Detection Results:")
        st.dataframe(df_uploaded.style.apply(lambda x: ["background-color: #ffe6e6" if x["Anomaly_Prediction"] == "Suspicious" else "" for i in x], axis=1))

        num_suspicious = (df_uploaded["Anomaly_Prediction"] == "Suspicious").sum()
        st.info(f"Detected {num_suspicious} suspicious transactions out of {len(df_uploaded)} processed records.")
else:
    st.warning("Model could not be loaded. Please ensure the anomaly detection model is trained and saved in the notebook.")
