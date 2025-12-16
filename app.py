import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time

st.markdown("""
<style>
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
}
button {
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

from preprocessing import (
    handle_missing_value,
    replace_outlier_with_median,
    normalize_data,
    standardize_data
)
from modeling import run_regression
from sklearn.metrics import mean_absolute_error


st.set_page_config(page_title="Prediksi Pertanian", layout="wide")
st.title("📈 Prediksi Data Menggunakan Regresi Linear")

# ================= UPLOAD FILE =================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset (CSV / Excel)",
    type=["csv", "xlsx"]
)

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

    # Jumlah Baris & Kolom
    st.write(f"Jumlah Baris: {df.shape[0]}")
    st.write(f"Jumlah Kolom: {df.shape[1]}")

    # Data Head
    st.write("Data Head (5 Data Teratas)")
    st.dataframe(df.head())

    # Data Tail
    st.write("Data Tail (5 Data Terakhir)")
    st.dataframe(df.tail())

    # Info Data (Tipe Data)
    st.write("Informasi Struktur Data")

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    st.code(info_str)

    # Statistik Deskriptif
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

    # Simulasi loading step-by-step
    status_text.text("📊 Menyiapkan data...")
    progress.progress(20)

    import time
    time.sleep(0.5)

    status_text.text("🧮 Melatih model regresi linear...")
    progress.progress(60)

    hasil = run_regression(df_ready, target=target_col)

    time.sleep(0.5)

    status_text.text("📈 Menghitung evaluasi model...")
    progress.progress(90)

    time.sleep(0.5)

    progress.progress(100)
    status_text.empty()
    progress.empty()

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
