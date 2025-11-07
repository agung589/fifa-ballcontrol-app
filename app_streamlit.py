import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === 1️⃣ Load model ===
with open('model_lr.pkl', 'rb') as f:
    model = pickle.load(f)

# === 2️⃣ Konfigurasi halaman ===
st.set_page_config(
    page_title="Prediksi Ball Control FIFA",
    page_icon="⚽",
    layout="centered"
)

# === 3️⃣ Styling Kustom ===
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0a0a0a;
            background-image: radial-gradient(circle at center, rgba(255,0,0,0.1) 0%, transparent 70%);
            color: white;
        }

        [data-testid="stSidebar"] {
            background-color: #111;
            border-right: 2px solid #d32f2f;
        }

        h1, h2, h3 {
            color: #ff4c4c;
            text-shadow: 0px 0px 12px rgba(255,0,0,0.6);
        }

        input[type=number] {
            background-color: #1a1a1a !important;
            color: white !important;
            border: 1px solid #d32f2f !important;
            border-radius: 6px !important;
        }

        div.stButton > button {
            background: linear-gradient(90deg, #b71c1c, #ff1744);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: 0.3s;
            box-shadow: 0 0 10px rgba(255,0,0,0.3);
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #ff5252, #ff1744);
            box-shadow: 0 0 25px rgba(255,0,0,0.6);
        }

        .prediksi-box {
            margin-top: 50px;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #ff4c4c;
            text-shadow: 0 0 20px rgba(255,0,0,0.8);
            border: 2px solid rgba(255,0,0,0.3);
            padding: 25px;
            border-radius: 15px;
            background-color: rgba(20,20,20,0.7);
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

# === 4️⃣ Judul utama ===
st.title("⚽ Prediksi Skill Ball Control Pemain FIFA")
st.markdown("Masukkan data pemain untuk memprediksi kemampuan **Ball Control** menggunakan model *Linear Regression*.")

# === 5️⃣ Sidebar input ===
st.sidebar.header("Masukkan Data Pemain")

feature_names = [
    'skill_dribbling', 
    'attacking_short_passing', 
    'attacking_crossing', 
    'skill_curve', 
    'goalkeeping_handling', 
    'attacking_finishing', 
    'attacking_volleys'
]

# Gunakan number_input (bukan slider)
user_input = []
for feature in feature_names:
    val = st.sidebar.number_input(
        label=feature.replace('_', ' ').title(),
        min_value=0,
        max_value=100,
        value=50
    )
    user_input.append(val)

# === 6️⃣ Tombol prediksi ===
if st.sidebar.button("Prediksi Ball Control 🎯"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]

    st.markdown(f"""
        <div class="prediksi-box">
            🎯 <br>Prediksi Ball Control:<br> 
            <span style="font-size:55px;">{prediction:.2f}</span>
        </div>
    """, unsafe_allow_html=True)

# === 7️⃣ Info tambahan ===
st.info("Model ini menggunakan Linear Regression hasil pelatihan pada dataset FIFA yang telah dibersihkan.")
