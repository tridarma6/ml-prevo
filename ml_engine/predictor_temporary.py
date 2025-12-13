import os
import json
import joblib
import zipfile
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

# Kolom yang wajib ada
NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

ORIGINAL_INPUT_COLS = [
    "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Target", "Failure Type"
]

# -----------------------
# Utility Helpers
# -----------------------
def _ensure_file(path, msg=None):
    if not os.path.exists(path):
        raise FileNotFoundError(msg or f"Required file not found: {path}")

def _zip_tabnet_folder_if_needed(tabnet_folder_path):
    # Logika zip TabNet folder
    if os.path.isfile(tabnet_folder_path) and tabnet_folder_path.lower().endswith(".zip"):
        return tabnet_folder_path, False

    folder_path = tabnet_folder_path
    if os.path.isdir(folder_path):
        mp = os.path.join(folder_path, "model_params.json")
        net = os.path.join(folder_path, "network.pt")
        if not (os.path.isfile(mp) and os.path.isfile(net)):
            raise FileNotFoundError(f"TabNet folder found but missing required files in {folder_path}")
        
        tmp_zip = os.path.join(folder_path, "..", os.path.basename(folder_path.rstrip("/\\")) + ".zip")
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except Exception:
                pass
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, arcname=fname)
        return tmp_zip, True

    raise FileNotFoundError(f"TabNet model path {tabnet_folder_path} not found.")

# -----------------------
# Load Artifacts
# -----------------------
def load_artifacts(model_dir):
    # Langsung gunakan model_dir yang dikirim dari API
    if not os.path.exists(model_dir):
         raise FileNotFoundError(f"MODEL_DIR not found: {model_dir}")

    artifacts = {}
    
    # Path file model
    f_xgb = os.path.join(model_dir, "failure_type_xgb_ovr_fe_models.joblib")
    f_le = os.path.join(model_dir, "label_encoder_failure_type_ovr_fe.joblib")
    f_scaler = os.path.join(model_dir, "scaler_fe.joblib")
    f_meta = os.path.join(model_dir, "tabnet_anomaly_meta.joblib")
    f_preproc = os.path.join(model_dir, "tabnet_anomaly_preproc.joblib")
    
    tabnet_folder_candidate_zip = os.path.join(model_dir, "tabnet_anomaly_model.zip")
    tabnet_folder_candidate_dir = os.path.join(model_dir, "tabnet_anomaly_model")

    _ensure_file(f_xgb)
    _ensure_file(f_le)
    _ensure_file(f_scaler)
    _ensure_file(f_meta)
    _ensure_file(f_preproc)

    if not (os.path.isfile(tabnet_folder_candidate_zip) or os.path.isdir(tabnet_folder_candidate_dir)):
        raise FileNotFoundError("Missing TabNet model file/folder")

    # Load Joblib
    artifacts["xgb_models"] = joblib.load(f_xgb)
    artifacts["le_failtype"] = joblib.load(f_le)
    artifacts["scaler_fe"] = joblib.load(f_scaler)
    artifacts["tabnet_meta"] = joblib.load(f_meta)
    artifacts["tabnet_preproc"] = joblib.load(f_preproc)

    # Load TabNet
    to_clean_zip = None
    try:
        if os.path.isfile(tabnet_folder_candidate_zip):
            tabnet_zip_path = tabnet_folder_candidate_zip
        else:
            tabnet_zip_path, is_temp = _zip_tabnet_folder_if_needed(tabnet_folder_candidate_dir)
            if is_temp: to_clean_zip = tabnet_zip_path
        
        clf = TabNetClassifier()
        clf.load_model(tabnet_zip_path)
        artifacts["tabnet_model"] = clf
    finally:
        if to_clean_zip and os.path.exists(to_clean_zip):
            try: os.remove(to_clean_zip)
            except: pass

    return artifacts

# -----------------------
# Feature Engineering
# -----------------------
def fe(df, le_type=None):
    df = df.copy()
    for c in NUM_COLS:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "Type" in df.columns:
        df["Type_Encoded"] = pd.factorize(df["Type"].astype(str).fillna("NA"))[0]
    elif "Type_Encoded" not in df.columns:
        df["Type_Encoded"] = 0

    df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Torque_Speed_Product"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Energy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * df["Process temperature [K]"]
    df["Load_Factor"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"].replace({0: np.nan}) + 1)
    df["Load_Factor"] = df["Load_Factor"].fillna(0.0)
    df["Temp_Ratio"] = df["Process temperature [K]"] / (df["Air temperature [K]"].replace({0: np.nan}) + 1)
    df["Temp_Ratio"] = df["Temp_Ratio"].fillna(0.0)

    for col in NUM_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1 if not pd.isna(Q3 - Q1) else 0.0
        df[col + "_Outlier"] = 0 if IQR == 0 else ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).astype(int)

    return df

# -----------------------
# Prediction Logics
# -----------------------
def predict_anomaly(df, artifacts):
    tabnet = artifacts["tabnet_model"]
    preproc = artifacts["tabnet_preproc"]
    meta = artifacts["tabnet_meta"]
    scaler = preproc["scaler"]

    feature_cols = ["Product_ID_enc", "Type_Encoded", "Air temperature [K]", "Process temperature [K]", 
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    if "Product_ID_enc" not in df.columns:
        df["Product_ID_enc"] = pd.factorize(df["Product ID"].astype(str).fillna("NA"))[0] if "Product ID" in df.columns else 0

    for c in feature_cols:
        if c not in df.columns: df[c] = 0.0

    X_tab = df[feature_cols].astype(float).values
    X_scaled = scaler.transform(X_tab)
    probs = tabnet.predict_proba(X_scaled.astype(np.float32))[:, 1]

    df["anomaly_probability"] = probs
    threshold = meta.get("threshold", 0.5)
    df["is_anomaly"] = (probs >= threshold).astype(int)
    return df

def predict_failure_type(df_fe, artifacts):
    models = artifacts["xgb_models"]
    le = artifacts["le_failtype"]
    scaler = artifacts["scaler_fe"]
    
    # Feature columns list (sesuai kode asli)
    feature_cols = [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Type_Encoded", "Temp_Diff", "Torque_Speed_Product", "Energy", "Load_Factor", "Temp_Ratio",
        "Air temperature [K]_Outlier", "Process temperature [K]_Outlier", "Rotational speed [rpm]_Outlier",
        "Torque [Nm]_Outlier", "Tool wear [min]_Outlier"
    ]

    for c in feature_cols:
        if c not in df_fe.columns: df_fe[c] = 0.0

    X = scaler.transform(df_fe[feature_cols].values)
    
    # Logic prediksi failure type (OVR)
    proba_list = []
    n_classes = len(le.classes_)
    for i in range(n_classes):
        model = models.get(i) if isinstance(models, dict) else (models[i] if i < len(models) else None)
        if model is None:
            proba_list.append(np.zeros((X.shape[0],)))
        else:
            proba_list.append(model.predict_proba(X)[:, 1])

    proba_stack = np.column_stack(proba_list)

    # Boosting rules
    for i, cls in enumerate(le.classes_):
        if cls == "Random Failures": proba_stack[:, i] *= 50
        if cls == "Tool Wear Failure": proba_stack[:, i] *= 40

    thresholds_override = {"Random Failures": 0.003, "Tool Wear Failure": 0.0045}
    proba_override = proba_stack.copy()
    
    for i_row in range(proba_override.shape[0]):
        for idx, cls in enumerate(le.classes_):
            if cls in thresholds_override and proba_stack[i_row, idx] > thresholds_override[cls]:
                proba_override[i_row, idx] = 9999

    pred_idx = np.argmax(proba_override, axis=1)
    df_fe["predicted_failure_type"] = le.inverse_transform(pred_idx.astype(int))
    return df_fe

# -----------------------
# Main Runner Function
# -----------------------
def run_and_save_api(input_csv, output_json, model_dir):
    """
    Fungsi ini dipanggil oleh FastAPI.
    """
    artifacts = load_artifacts(model_dir)
    df = pd.read_csv(input_csv)
    
    df_fe = fe(df)
    df_with_anom = predict_anomaly(df_fe, artifacts)
    df_final = predict_failure_type(df_fe, artifacts)
    
    df_final["anomaly_probability"] = df_with_anom["anomaly_probability"]
    df_final["is_anomaly"] = df_with_anom["is_anomaly"]

    recommendation_map = {
        "No Failure": "Tidak perlu tindakan. Jadwalkan pemeliharaan rutin sesuai SOP.",
        "Heat Dissipation Failure": "Periksa sistem pendingin/ventilasi, bersihkan kipas dan saluran udara.",
        "Power Failure": "Periksa suplai listrik, kabel, konektor, serta UPS/stabilizer.",
        "Overstrain Failure": "Kurangi beban operasional; lakukan balancing dan cek komponen mekanik.",
        "Random Failures": "Lakukan inspeksi menyeluruh; cek log operasi dan sensor.",
        "Tool Wear Failure": "Ganti atau asah tool; cek parameter pemotongan."
    }
    
    df_final["recommendation"] = df_final["predicted_failure_type"].map(
        lambda x: recommendation_map.get(x, "Lakukan inspeksi manual.")
    )

    # Simpan JSON Compact
    result_cols = ["anomaly_probability", "is_anomaly", "predicted_failure_type", "recommendation"]
    cols_to_keep = [c for c in ORIGINAL_INPUT_COLS if c in df_final.columns] + result_cols
    
    compact_df = df_final[cols_to_keep].copy()
    records = compact_df.to_dict(orient="records")
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    return len(compact_df)