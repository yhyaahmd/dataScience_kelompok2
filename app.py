import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time

# ================= PAGE CONFIG (WAJIB PALING ATAS) =================
st.set_page_config(
    page_title="Prediksi Pertanian",
    layout="wide"
)

# ================= GLOBAL FONT (POPPINS) =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ================= IMPORT MODULE =================
from preprocessing import (
    handle_missing_value,
    replace_outlier_with_median,
    normalize_data,
    standardize_data
)
from modeling import run_regression
from sklearn.metrics import mean_absolute_error

# ============== Buat Banner ==============
carousel_items = [
    {
        "title": "Analisis Pertanian Cerdas",
        "text": "Menggunakan Data Science untuk hasil panen maksimal",
        "interval": None,
        "img": "https://images.unsplash.com/photo-1625246333195-09d9b630dc93?q=80&w=1920&auto=format&fit=crop"
    },
    {
        "title": "Prediksi Akurat",
        "text": "Algoritma Regresi Linear untuk estimasi produksi",
        "interval": None,
        "img": "https://images.unsplash.com/photo-1560493676-04071c5f467b?q=80&w=1920&auto=format&fit=crop"
    },
    {
        "title": "Teknologi & Alam",
        "text": "Integrasi teknologi modern dalam agrikultur",
        "interval": None,
        "img": "https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=1920&auto=format&fit=crop"
    }
]

# Menampilkan Carousel
# Container digunakan agar carousel tidak terlalu lebar jika di layar ultra-wide
with st.container():
    carousel(items=carousel_items)

# ================= TITLE =================
st.title("📈 Prediksi Data Menggunakan Regresi Linear")

# ================= UPLOAD FILE =================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset (CSV / Excel)",
    type=["csv", "xlsx"]
)

# ================= MAIN FLOW =================
if uploaded_file:

    # ================= LOAD DATA =================
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Preview Data")
    st.dataframe(df)

    # ================= DATA UNDERSTANDING =================
    st.subheader("🔍 Data Understanding")

    st.write(f"Jumlah Baris: {df.shape[0]}")
    st.write(f"Jumlah Kolom: {df.shape[1]}")

    st.write("Data Head (5 Data Teratas)")
    st.dataframe(df.head())

    st.write("Data Tail (5 Data Terakhir)")
    st.dataframe(df.tail())

    st.write("Informasi Struktur Data")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue())

    st.write("Statistik Deskriptif")
    st.dataframe(df.describe())

    # ================= PREPROCESSING =================
    st.subheader("🧹 Data Preprocessing")

    # Missing Value
    df_clean, info_missing = handle_missing_value(df)

    st.markdown("📌 Penanganan Missing Value")
    st.markdown("🔹 Jumlah Missing Value Sebelum")
    st.dataframe(info_missing["missing_before"])

    st.markdown("🔹 Detail Penanganan Missing Value")
    if not info_missing["handled_columns"].empty:
        st.dataframe(info_missing["handled_columns"])
    else:
        st.info("Tidak ditemukan missing value pada seluruh kolom data.")

    st.markdown("🔹 Jumlah Missing Value Sesudah")
    st.dataframe(info_missing["missing_after"])

    # Outlier
    show_plot = st.checkbox("Tampilkan Boxplot Outlier")
    df_no_outlier, laporan_outlier, fig = replace_outlier_with_median(
        df_clean, show_plot=show_plot
    )

    st.markdown("📌 Penanganan Outlier")
    st.dataframe(laporan_outlier)

    if fig:
        st.pyplot(fig)

    # ================= SCALING =================
    metode = st.radio(
        "Pilih Metode Skala Data:",
        ["Normalisasi", "Standarisasi"]
    )

    if metode == "Normalisasi":
        df_ready = normalize_data(df_no_outlier)
    else:
        df_ready = standardize_data(df_no_outlier)

    st.markdown("📊 Data Setelah Preprocessing")
    st.dataframe(df_ready.head())

    # ================= TARGET =================
    numeric_cols = df_ready.select_dtypes(include="number").columns

    target_col = st.selectbox(
        "🎯 Pilih Kolom Target (Variabel Y - Numerik)",
        numeric_cols
    )

    # ================= MODELING =================
    if st.button("🚀 Jalankan Regresi Linear"):

        progress = st.progress(0)
        status_text = st.empty()

        status_text.text("📊 Menyiapkan data...")
        progress.progress(20)
        time.sleep(0.5)

        status_text.text("🧮 Melatih model regresi linear...")
        progress.progress(60)
        hasil = run_regression(df_ready, target=target_col)
        time.sleep(0.5)

        status_text.text("📈 Menghitung evaluasi model...")
        progress.progress(90)
        time.sleep(0.5)

        progress.progress(100)
        progress.empty()
        status_text.empty()

        st.success("✅ Model berhasil dijalankan!")

        # ================= EVALUASI =================
        st.subheader("📊 Evaluasi Model")
        st.write(f"**R² Score:** {hasil['r2']:.4f}")
        st.write(f"**MSE:** {hasil['mse']:.4f}")
        st.write(f"**RMSE:** {hasil['rmse']:.4f}")
        st.write(f"**MAE:** {hasil['mae']:.4f}")

        # ================= PREDIKSI =================
        st.subheader("📈 Hasil Prediksi")
        df_pred = pd.DataFrame({
            "Aktual": hasil["y_test"].values,
            "Prediksi": hasil["y_pred"]
        })
        st.dataframe(df_pred)

        # ================= VISUAL =================
        st.subheader("📉 Visualisasi Aktual vs Prediksi")
        fig2, ax = plt.subplots()

        sns.scatterplot(
            x=df_pred["Aktual"],
            y=df_pred["Prediksi"],
            hue=df_pred["Prediksi"] >= df_pred["Aktual"],
            palette={True: "blue", False: "orange"},
            legend=False,
            ax=ax
        )

        ax.set_xlabel("Nilai Aktual")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title("Scatter Plot Aktual vs Prediksi", fontweight="bold")

        st.pyplot(fig2)

        # ================= KOEFISIEN =================
        st.subheader("📌 Koefisien Regresi")
        st.dataframe(hasil["coef"])

else:
    st.info("⬆️ Silakan upload dataset terlebih dahulu untuk memulai.")
