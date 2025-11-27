import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Pembelian Windows (Random Forest)", layout="centered")

# 2. Muat Model RF
@st.cache_resource 
def load_model():
    try:
        # Nama file sudah diganti menjadi rf_pipeline.pkl
        model = joblib.load('data.pkl') 
        return model
    except FileNotFoundError:
        st.error("File model 'data.pkl' tidak ditemukan. Harap jalankan train_model.ipynb terlebih dahulu.")
        st.stop()

pipeline_loaded = load_model()

st.title("Prediksi Pembelian Lisensi Windows")
st.subheader("Model Klasifikasi yang Digunakan: Random Forest")
st.subheader("Kelompok 1:")
st.markdown("Yogi Ario Pratama | 2313020004")
st.markdown("Shandy P | 2313020069")
st.markdown("Achmad Fardani | 232302008")

st.markdown("---")
st.markdown("### Masukkan Data Pelanggan")

PEKERJAAN_OPTIONS = ['Akuntan', 'Dokter', 'Guru', 'Karyawan Swasta', 'Pengusaha', 'Lainnya', 'Hotel Staff', 'Engineer', 'Designer', 'Content Creator'] 
col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("1. Usia", min_value=18, max_value=100, value=30, step=1, help="Masukkan usia pelanggan (angka bulat)")
    
    penghasilan = st.number_input("2. Penghasilan Bulanan (Juta)", min_value=1.0, max_value=100.0, value=15.0, step=0.1, help="Contoh: 15.0")
    
with col2:
    pekerjaan = st.selectbox("3. Pekerjaan", PEKERJAAN_OPTIONS, help="Pilih jenis pekerjaan pelanggan")
    
    kemampuan_teknologi = st.number_input("4. Kemampuan Teknologi (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, help="Nilai antara 0.0 (rendah) dan 1.0 (tinggi)")

kebutuhan_bisnis = st.number_input("5. Kebutuhan Bisnis (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, help="Nilai antara 0.0 (rendah) dan 1.0 (tinggi)")


if st.button("PREDIKSI HASIL", type="primary"):
    data_input = pd.DataFrame({
        'Usia': [usia],
        'Pekerjaan': [pekerjaan],
        'Penghasilan_Bulanan_Juta': [penghasilan],
        'Kemampuan_Teknologi': [kemampuan_teknologi],
        'Kebutuhan_Bisnis': [kebutuhan_bisnis]
    })
    
    hasil_prediksi = pipeline_loaded.predict(data_input)[0]
    
    st.markdown("---")
    st.subheader("Hasil Prediksi Model Random Forest:")

    if hasil_prediksi == 1:
        st.success(f"**PELANGGAN DIPREDIKSI AKAN BELI LISENSI WINDOWS!**")
    else:
        st.warning(f"**PELANGGAN DIPREDIKSI TIDAK AKAN BELI LISENSI WINDOWS.**")
        

    st.info("Prediksi didasarkan pada model Random Forest yang telah dilatih.")
