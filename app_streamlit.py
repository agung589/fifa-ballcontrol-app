import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# === 1️⃣ Load model yang sudah dilatih ===
with open('model_lr.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("⚽ Prediksi Skill Ball Control Pemain FIFA")
st.write("Aplikasi ini memprediksi nilai *Ball Control* berdasarkan skill-skill menyerang menggunakan model Linear Regression.")

# === 2️⃣ Sidebar input pengguna ===
st.sidebar.header("Masukkan Data Pemain")

# Daftar fitur (HARUS sesuai urutan training waktu di notebook)
feature_names = [
    'skill_dribbling', 
    'attacking_short_passing', 
    'attacking_crossing', 
    'skill_curve', 
    'goalkeeping_handling', 
    'attacking_finishing', 
    'attacking_volleys'
]

# Input slider untuk setiap fitur
user_input = []
for feature in feature_names:
    val = st.sidebar.slider(
        label=feature.replace('_', ' ').title(),
        min_value=0,
        max_value=100,
        value=50
    )
    user_input.append(val)

# === 3️⃣ Prediksi ===
input_array = np.array([user_input])
prediction = model.predict(input_array)[0]

st.sidebar.markdown(f"🎯 **Prediksi Ball Control:** `{prediction:.2f}`")

# === 4️⃣ (Opsional) Visualisasi sederhana ===
st.subheader("📈 Distribusi Prediksi")
fig, ax = plt.subplots()
ax.bar(['Prediksi Ball Control'], [prediction], color='skyblue')
ax.set_ylim(0, 100)
st.pyplot(fig)

# === 5️⃣ Info tambahan ===
st.info("Model ini menggunakan Linear Regression hasil pelatihan pada dataset FIFA yang telah dibersihkan.")
