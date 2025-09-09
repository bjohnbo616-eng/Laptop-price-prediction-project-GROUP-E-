import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and feature names
model = joblib.load('laptop_price.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature.joblib')

# Load the dataset to get unique values - handle merge conflict in CSV
try:
    # Try reading normally first
    df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
    # Check if we have merge conflict markers
    if df.columns[0].startswith('<'):
        raise ValueError("Merge conflict detected")
except:
    # Read the file and skip merge conflict lines
    with open('laptop_price.csv', 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
    
    # Filter out merge conflict markers
    clean_lines = []
    skip = False
    for line in lines:
        if line.startswith('<<<<<<< HEAD') or line.startswith('=======') or line.startswith('>>>>>>> '):
            skip = not skip if line.startswith('=======') else skip
            continue
        if not skip:
            clean_lines.append(line)
    
    # Write to a temporary clean file and read it
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='ISO-8859-1') as tmp:
        tmp.writelines(clean_lines)
        tmp_path = tmp.name
    
    df = pd.read_csv(tmp_path, encoding='ISO-8859-1')
    os.unlink(tmp_path)  # Clean up temp file

# Get unique values for categorical inputs (check which column names exist)
if 'company' in df.columns:
    companies = sorted(df['company'].unique())
    type_names = sorted(df['typename'].unique())
elif 'Company' in df.columns:
    companies = sorted(df['Company'].unique())
    type_names = sorted(df['TypeName'].unique())
else:
    st.error("Could not find company column in the dataset")
    st.stop()

storage_types = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']

# Common screen resolutions
resolutions = [
    '1366x768', '1600x900', '1920x1080', '2560x1440', '2560x1600',
    '2880x1800', '3200x1800', '3840x2160'
]

# Streamlit app
st.title('Laptop Price Prediction App')
st.write('Enter the laptop specifications to predict the price in Euros.')

# Debug info
st.sidebar.write("Debug Info:")
st.sidebar.write(f"Dataset shape: {df.shape}")
st.sidebar.write(f"Columns: {list(df.columns)}")

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