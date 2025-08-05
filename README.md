# Customer Churn Prediction App

A Streamlit web application that predicts customer churn probability using an Artificial Neural Network (ANN) model.

## Features

- Interactive web interface for customer churn prediction
- Real-time probability calculation
- User-friendly input forms
- Visual progress indicators
- Error handling and validation

## Files Structure

```
├── app.py                          # Main Streamlit application
├── model.h5                        # Trained ANN model
├── scaler.pkl                      # StandardScaler for feature scaling
├── label_encoder_gender.pkl        # LabelEncoder for gender encoding
├── onehot_encoder_geo.pkl          # OneHotEncoder for geography encoding
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies
├── .streamlit/config.toml         # Streamlit configuration
└── Churn_Modelling.csv            # Original dataset
```

## Deployment Instructions

### For Streamlit Cloud:

1. **Upload your files** to a GitHub repository
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Set the main file path to `app.py`
   - Deploy

### For Local Development:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

### For Other Platforms (Heroku, Railway, etc.):

1. Ensure all files are in the root directory
2. The platform will automatically detect and install dependencies from `requirements.txt`
3. Set the command to: `streamlit run app.py`

## Model Information

The application uses a trained Artificial Neural Network with the following features:

- **Input Features**: Credit Score, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary, Geography
- **Output**: Churn probability (0-1)
- **Threshold**: 0.5 (above which customer is predicted to churn)

## Troubleshooting

### Common Issues:

1. **"No module named 'tensorflow'"**:
   - Ensure `requirements.txt` is present and contains tensorflow
   - Try using specific versions: `tensorflow==2.15.0`

2. **Model loading errors**:
   - Verify all `.pkl` and `.h5` files are in the root directory
   - Check file permissions

3. **Deployment platform issues**:
   - Some platforms may have memory limitations
   - Consider using lighter model formats if needed

## Dependencies

- tensorflow==2.15.0
- pandas==2.1.4
- numpy==1.24.3
- scikit-learn==1.3.2
- matplotlib==3.8.2
- streamlit==1.29.0 
