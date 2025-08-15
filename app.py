import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------- Load model and data -----------------
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# ----------------- Page Configuration -----------------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .main {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        @media only screen and (max-width: 600px) {
            .block-container {
                padding: 1rem !important;
            }
        }
        h1 {
            color: #2e86de;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")
st.subheader("Enter Laptop Specifications")

# ----------------- Layout -----------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox('Brand', df['Company'].unique())
    type_name = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 32])
    weight = st.number_input('Weight (in Kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips_val = st.selectbox('IPS Display', ['No', 'Yes'])

with col2:
    screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1)
    resolution = st.selectbox(
        'Screen Resolution', 
        ['1920x1080', '1366x768', '1600x900', '3840x2160', 
         '2560x1600', '2736x1824', '2560x1440']
    )
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
    os = st.selectbox('Operating System', df['os'].unique())

# ----------------- Prediction -----------------
if st.button('💰 Predict Price'):
    # Convert Yes/No to binary
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips_val == 'Yes' else 0

    # Calculate PPI
    try:
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
    except Exception:
        st.error("⚠️ Error in calculating PPI. Please check resolution and screen size.")
        ppi = 0

    # Prepare input DataFrame
    input_df = pd.DataFrame([{
        'Company': brand,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'IPS': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    # Predict
    try:
        predicted_price = int(np.exp(pipe.predict(input_df)[0]))
        st.success(f"💸 Predicted Laptop Price: ₹ {predicted_price:,}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Made with ❤️ by Ankit</h5>", unsafe_allow_html=True)
