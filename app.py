import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load encoders and scaler with error handling
@st.cache_resource
def load_encoders():
    try:
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading encoders: {str(e)}")
        return None, None, None

# Load all resources
model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# Check if all resources are loaded successfully
if model is None or label_encoder_gender is None or onehot_encoder_geo is None or scaler is None:
    st.error("Failed to load required model files. Please ensure all files are present.")
    st.stop()

## streamlit app
st.title('Customer Churn Prediction 📊')
st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    # User input
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 35)
    balance = st.number_input('Balance', value=0.0)
    credit_score = st.number_input('Credit Score', value=600, min_value=300, max_value=850)

with col2:
    st.subheader("Account Details")
    estimated_salary = st.number_input('Estimated Salary', value=50000.0)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prediction button
if st.button('Predict Churn Probability', type="primary"):
    try:
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create a progress bar for visualization
        st.progress(prediction_proba)
        
        # Display probability with color coding
        if prediction_proba > 0.5:
            st.error(f'⚠️ Churn Probability: {prediction_proba:.2%}')
            st.error('🚨 The customer is likely to churn.')
        else:
            st.success(f'✅ Churn Probability: {prediction_proba:.2%}')
            st.success('✅ The customer is not likely to churn.')
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Add some helpful information
with st.expander("ℹ️ About this Model"):
    st.write("""
    This model predicts customer churn probability based on various customer attributes including:
    - Demographics (Age, Gender, Geography)
    - Financial information (Credit Score, Balance, Estimated Salary)
    - Account details (Tenure, Number of Products, Credit Card status)
    - Engagement (Active Member status)
    
    The model uses an Artificial Neural Network trained on historical customer data.
    """)
