import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ===============================
# 1. MISSING VALUE
# ===============================
def handle_missing_value(df):
    df = df.copy()
    info = {}

    # Missing value sebelum
    missing_before = df.isnull().sum()
    info["missing_before"] = missing_before

    handled_columns = []

    # Tangani missing value SEMUA kolom menggunakan MODUS
    for col in df.columns:
        jumlah_missing = missing_before[col]

        if jumlah_missing > 0:
            # Hitung modus (aman untuk numerik & kategorikal)
            modus_series = df[col].mode()

            if not modus_series.empty:
                modus = modus_series[0]
                df[col] = df[col].fillna(modus)

                handled_columns.append({
                    "Kolom": col,
                    "Tipe Data": str(df[col].dtype),
                    "Metode": "Modus",
                    "Nilai Pengganti": modus,
                    "Jumlah Missing Value": jumlah_missing
                })

    # Missing value sesudah
    info["missing_after"] = df.isnull().sum()
    info["handled_columns"] = pd.DataFrame(handled_columns)

    return df, info


# ===============================
# 2. HITUNG JUMLAH OUTLIER (IQR)
# ===============================
def count_outlier_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2.5 * IQR
    upper = Q3 + 2.5 * IQR
    return ((df[col] < lower) | (df[col] > upper)).sum()


# ===============================
# 3. GANTI OUTLIER DENGAN MEDIAN
# ===============================
def replace_outlier_with_median(df, show_plot=False):
    df = df.copy()

    # Ambil kolom numerik kecuali Tahun
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = [c for c in num_cols if c != "Tahun"]

    laporan = []

    # Proses tiap kolom numerik
    for col in num_cols:
        outlier_before = count_outlier_iqr(df, col)

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 2.5 * IQR
        upper = Q3 + 2.5 * IQR
        median = df[col].median()

        # Ganti outlier dengan median
        df[col] = np.where(
            (df[col] < lower) | (df[col] > upper),
            median,
            df[col]
        )

        outlier_after = count_outlier_iqr(df, col)

        laporan.append({
            "Kolom": col,
            "Outlier Sebelum": outlier_before,
            "Outlier Sesudah": outlier_after
        })

    laporan_df = pd.DataFrame(laporan)

    # ===============================
    # BOXPLOT (RAPI & TIDAK KETIMPA)
    # ===============================
    fig = None
    if show_plot and len(num_cols) > 0:
        fig, ax = plt.subplots(
            len(num_cols),
            1,
            figsize=(10, 5 * len(num_cols)),
            constrained_layout=True
        )

        if len(num_cols) == 1:
            ax = [ax]

        for i, col in enumerate(num_cols):
            sns.boxplot(x=df[col], ax=ax[i], whis=2.5)
            ax[i].set_title(
                f"Boxplot {col} (Metode: Median)",
                fontsize=12,
                pad=15
            )
            ax[i].set_xlabel(col)

    return df, laporan_df, fig


# ===============================
# 4. NORMALISASI
# ===============================
def normalize_data(df):
    df = df.copy()
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ===============================
# 5. STANDARISASI
# ===============================
def standardize_data(df):
    df = df.copy()
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
