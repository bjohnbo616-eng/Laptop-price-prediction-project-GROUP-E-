import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and feature names
model = joblib.load('laptop_price.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature.joblib')

# Load the dataset to get unique values
df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')

# Get unique values for categorical inputs
companies = sorted(df['Company'].unique())
type_names = sorted(df['TypeName'].unique())
storage_types = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']

# Common screen resolutions
resolutions = [
    '1366x768', '1600x900', '1920x1080', '2560x1440', '2560x1600',
    '2880x1800', '3200x1800', '3840x2160'
]

# Streamlit app
st.title('Laptop Price Prediction App')
st.write('Enter the laptop specifications to predict the price in Euros.')

# Input widgets
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Company', companies)
    type_name = st.selectbox('Type Name', type_names)
    # Use a dropdown for screen size instead of a slider for better UX
    inches_options = [11.6, 12.0, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3]
    inches = st.selectbox('Screen Size (Inches)', inches_options, index=inches_options.index(15.6))
    ram = st.selectbox('RAM (GB)', [4, 8, 16, 32, 64])
    weight = st.number_input('Weight (kg)', 0.5, 5.0, 2.0, 0.1)

with col2:
    resolution = st.selectbox('Screen Resolution', resolutions)
    touchscreen = st.checkbox('Touchscreen')
    ips_panel = st.checkbox('IPS Panel')
    storage_size = st.number_input('Storage Size (GB)', 32, 2000, 256)
    storage_type = st.selectbox('Storage Type', storage_types)

# Predict button
if st.button('Predict Price'):
    # Parse resolution
    width, height = map(int, resolution.split('x'))
    pixels = width * height

    # Create input dataframe
    input_data = pd.DataFrame({
        'Inches': [inches],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [1 if touchscreen else 0],
        'IPS_Panel': [1 if ips_panel else 0],
        'Pixels': [pixels],
        'Storage_Size_GB': [storage_size]
    })

    # One-hot encode categorical
    for comp in companies[1:]:  # Skip first for drop_first
        input_data[f'Company_{comp}'] = [1 if company == comp else 0]
    for typ in type_names[1:]:
        input_data[f'TypeName_{typ}'] = [1 if type_name == typ else 0]
    for stor in storage_types[1:]:
        input_data[f'Storage_Type_{stor}'] = [1 if storage_type == stor else 0]

    # Ensure all features are present
    for feat in feature_names:
        if feat not in input_data.columns:
            input_data[feat] = 0

    # Reorder columns
    input_data = input_data[feature_names]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f'Predicted Price: €{prediction:.2f}')

    # Optional: Show feature contributions
    st.write('### Feature Contributions')
    coeffs = model.coef_
    contributions = input_scaled[0] * coeffs
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contributions
    }).sort_values('Contribution', ascending=False)
    st.dataframe(contrib_df.head(10))
