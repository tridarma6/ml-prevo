import os
import json
import joblib
import zipfile
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

# =========================================================
# CONFIG & CONSTANTS
# =========================================================
NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

ORIGINAL_INPUT_COLS = [
    "UDI", "Product ID", "Type", 
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", 
    "Target", "Failure Type"
]

# =========================================================
# UTILITIES
# =========================================================
def _ensure_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

def _ensure_tabnet_zip(model_dir):
    """
    Logika perbaikan dari teman Anda:
    Memastikan file zip TabNet ada. Jika hanya ada foldernya,
    fungsi ini akan men-zip file yang dibutuhkan (model_params.json & network.pt).
    """
    zip_path = os.path.join(model_dir, "tabnet_anomaly_model.zip")
    folder_path = os.path.join(model_dir, "tabnet_anomaly_model")

    # 1. Jika zip sudah ada, langsung return path-nya
    if os.path.isfile(zip_path):
        return zip_path

    # 2. Jika zip tidak ada, cek apakah foldernya ada
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"TabNet model zip/folder not found inside {model_dir}")

    mp = os.path.join(folder_path, "model_params.json")
    net = os.path.join(folder_path, "network.pt")
    
    # 3. Pastikan isi foldernya lengkap
    if not (os.path.isfile(mp) and os.path.isfile(net)):
        raise FileNotFoundError("TabNet folder missing model_params.json or network.pt")

    # 4. Buat zip sementara secara otomatis
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(mp, arcname="model_params.json")
            zf.write(net, arcname="network.pt")
    except Exception as e:
        raise RuntimeError(f"Failed to create TabNet zip: {e}")

    return zip_path

# =========================================================
# LOAD ARTIFACTS
# =========================================================
def load_artifacts(model_dir):
    # Validasi folder
    _ensure_file(model_dir)

    # Load Joblib artifacts
    # Menggunakan os.path.join agar path tidak hardcoded
    artifacts = {
        "xgb_models": joblib.load(os.path.join(model_dir, "failure_type_xgb_ovr_fe_models.joblib")),
        "le_failtype": joblib.load(os.path.join(model_dir, "label_encoder_failure_type_ovr_fe.joblib")),
        "scaler_fe": joblib.load(os.path.join(model_dir, "scaler_fe.joblib")),
        "tabnet_meta": joblib.load(os.path.join(model_dir, "tabnet_anomaly_meta.joblib")),
        "tabnet_preproc": joblib.load(os.path.join(model_dir, "tabnet_anomaly_preproc.joblib")),
    }

    # Load TabNet Model dengan helper baru
    tabnet_zip = _ensure_tabnet_zip(model_dir)
    clf = TabNetClassifier()
    # Pytorch TabNet load_model butuh path ke file .zip
    clf.load_model(tabnet_zip)
    artifacts["tabnet_model"] = clf

    return artifacts

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def fe(df):
    df = df.copy()

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # PENTING: Menggunakan nama kolom 'Type_enc' (sesuai kode teman Anda)
    # Kode lama menggunakan 'Type_Encoded', ini harus konsisten dengan training.
    if "Product ID" in df.columns:
        df["Product_ID_enc"] = pd.factorize(df["Product ID"].astype(str))[0]
    else:
        df["Product_ID_enc"] = 0

    if "Type" in df.columns:
        df["Type_enc"] = pd.factorize(df["Type"].astype(str))[0]
    else:
        df["Type_enc"] = 0

    # Feature Engineering Fisika
    df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Torque_Speed_Product"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Energy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * df["Process temperature [K]"]
    df["Load_Factor"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1)
    df["Temp_Ratio"] = df["Process temperature [K]"] / (df["Air temperature [K]"] + 1)

    # Outlier Detection (IQR)
    for col in NUM_COLS:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[f"{col}_Outlier"] = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).astype(int)

    return df

# =========================================================
# ANOMALY PREDICTION (TABNET)
# =========================================================
def predict_anomaly(df, artifacts):
    scaler = artifacts["tabnet_preproc"]["scaler"]
    threshold = artifacts["tabnet_meta"].get("threshold", 0.5)
    tabnet = artifacts["tabnet_model"]

    feature_cols = [
        "Product_ID_enc",
        "Type_enc", 
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    # Pastikan kolom ada
    for c in feature_cols:
        if c not in df.columns: df[c] = 0.0

    X = scaler.transform(df[feature_cols].astype(float).values)
    
    # Predict Proba
    probs = tabnet.predict_proba(X.astype(np.float32))[:, 1]

    df["anomaly_probability"] = probs
    df["is_anomaly"] = (probs >= threshold).astype(int)
    return df

# =========================================================
# FAILURE TYPE PREDICTION (XGB OVR)
# =========================================================
def predict_failure_type(df, artifacts):
    models = artifacts["xgb_models"]
    le = artifacts["le_failtype"]
    scaler = artifacts["scaler_fe"]

    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_enc",  # Perhatikan ini Type_enc
        "Temp_Diff",
        "Torque_Speed_Product",
        "Energy",
        "Load_Factor",
        "Temp_Ratio",
        "Air temperature [K]_Outlier",
        "Process temperature [K]_Outlier",
        "Rotational speed [rpm]_Outlier",
        "Torque [Nm]_Outlier",
        "Tool wear [min]_Outlier"
    ]
    
    # Pastikan kolom ada
    for c in feature_cols:
        if c not in df.columns: df[c] = 0.0

    X = scaler.transform(df[feature_cols].values)

    # Loop models (Logic teman Anda lebih rapi di sini)
    proba = []
    for i in range(len(le.classes_)):
        model = models.get(i)
        proba.append(model.predict_proba(X)[:, 1] if model else np.zeros(len(X)))

    proba = np.column_stack(proba)

    # Boosting Rules (Heuristic Adjustment)
    for i, cls in enumerate(le.classes_):
        if cls == "Random Failures":
            proba[:, i] *= 50
        if cls == "Tool Wear Failure":
            proba[:, i] *= 40

    # Override thresholds
    override = {"Random Failures": 0.003, "Tool Wear Failure": 0.0045}
    boosted = proba.copy()
    
    for i, cls in enumerate(le.classes_):
        if cls in override:
            # Jika probabilitas > threshold override, paksa jadi sangat tinggi
            boosted[proba[:, i] > override[cls], i] = 9999

    pred_idx = np.argmax(boosted, axis=1)
    df["predicted_failure_type"] = le.inverse_transform(pred_idx)

    return df

# =========================================================
# API BRIDGE (Function called by main.py)
# =========================================================
def run_and_save_api(input_csv, output_json, model_dir):
    """
    Fungsi ini dipanggil oleh FastAPI (main.py).
    Menggabungkan semua langkah: Load -> FE -> Predict -> Save.
    """
    # 1. Load Artifacts
    # Kita passing model_dir dari parameter function, bukan hardcode
    artifacts = load_artifacts(model_dir)

    # 2. Read CSV
    df = pd.read_csv(input_csv)

    # 3. Feature Engineering
    df = fe(df)

    # 4. Predict Anomaly
    df = predict_anomaly(df, artifacts)

    # 5. Predict Failure Type
    df = predict_failure_type(df, artifacts)

    # 6. Map Recommendation
    recommendation_map = {
        "No Failure": "Tidak perlu tindakan. Jadwalkan pemeliharaan rutin sesuai SOP.",
        "Heat Dissipation Failure": "Periksa sistem pendingin/ventilasi.",
        "Power Failure": "Periksa suplai listrik dan konektor.",
        "Overstrain Failure": "Kurangi beban operasional.",
        "Random Failures": "Lakukan inspeksi menyeluruh dan tingkatkan monitoring.",
        "Tool Wear Failure": "Ganti atau asah tool."
    }
    
    # Gunakan .get agar tidak crash jika ada failure type tak dikenal
    df["recommendation"] = df["predicted_failure_type"].map(
        lambda x: recommendation_map.get(x, "Lakukan inspeksi manual.")
    )

    # 7. Siapkan JSON (Compact Format)
    result_cols = ["anomaly_probability", "is_anomaly", "predicted_failure_type", "recommendation"]
    
    # Gabungkan kolom input asli dengan hasil prediksi
    cols_to_keep = [c for c in ORIGINAL_INPUT_COLS if c in df.columns] + result_cols
    
    compact_df = df[cols_to_keep].copy()
    records = compact_df.to_dict(orient="records")

    # 8. Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Return jumlah data untuk response API
    return len(compact_df)