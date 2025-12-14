import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# import predictor.py
from ml_engine.predictor import run_and_save_api

app = FastAPI(
    title="Predictive Maintenance API",
    description="API untuk Upload CSV, Run Model, dan Baca Hasil Prediksi.",
    version="2.0",
)

# CORS, kasi akses frontend nya
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Konfigurasi Path
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Path file
CSV_PATH = os.path.join(DATA_DIR, "predictive_maintenance.csv")
JSON_PATH = os.path.join(DATA_DIR, "predictive_maintenance_results.json")

# Variabel Global untuk simpan di memori
MACHINES = []


def load_data_to_memory():
    """Fungsi helper untuk memuat/refresh data JSON ke variabel global"""
    global MACHINES
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                MACHINES = json.load(f)
            print(f"[INFO] Data loaded: {len(MACHINES)} records.")
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {e}")
            MACHINES = []
    else:
        print("[WARN] JSON file not found. Waiting for model run.")
        MACHINES = []


# Load data saat aplikasi pertama kali jalan
@app.on_event("startup")
async def startup_event():
    # Pastikan folder data ada
    os.makedirs(DATA_DIR, exist_ok=True)
    load_data_to_memory()


# -------------------------
# Load JSON saat startup
# -------------------------
# with open("predictive_maintenance_results.json", "r") as f:
#     MACHINES = json.load(f)


# -------------------------
#   API UPLOAD DATASET
# -------------------------
@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        # Simpan file ke folder data (overwrite yang lama)
        with open(CSV_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully.",
            "path": CSV_PATH,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# -------------------------
#   API RUN MODEL
# -------------------------
@app.post("/model/run")
async def trigger_model():
    # Cek apakah CSV dan Model ada
    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=404, detail="Dataset CSV not found. Please upload first."
        )

    if not os.path.exists(MODELS_DIR):
        raise HTTPException(status_code=500, detail="Models directory not found.")

    try:
        # Jalankan fungsi ML (dari predictor.py)
        count = run_and_save_api(
            input_csv=CSV_PATH, output_json=JSON_PATH, model_dir=MODELS_DIR
        )

        # Refresh data di memori agar endpoint GET /machines/all langsung dapat data baru
        load_data_to_memory()

        return {
            "status": "success",
            "message": "Model executed successfully.",
            "processed_records": count,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")


# -------------------------
# GET: Semua mesin
# -------------------------
# @app.get("/machines/all")
# def get_all():
#     return MACHINES


# -------------------------
# GET: Mesin risiko tertinggi (top-1)
# -------------------------
@app.get("/machines/highest_risk")
def highest_risk():
    sorted_data = sorted(MACHINES, key=lambda x: x["anomaly_probability"], reverse=True)
    return sorted_data[0]


# -------------------------
# GET: Top N mesin paling berisiko optional dengan parameter failure type
# -------------------------
@app.get("/machines/top/{n}")
def get_top_n(n: int = 5, failure_type: Optional[str] = None):
    data = MACHINES

    if failure_type:
        data = [
            m
            for m in data
            if failure_type.lower() == m.get("predicted_failure_type", "").lower()
        ]

    sorted_data = sorted(data, key=lambda x: x["anomaly_probability"], reverse=True)

    return sorted_data[:n]


# -------------------------
# GET: Top N mesin resiko paling rendah (optional dengan parameter failure type)
# -------------------------
@app.get("/machines/bottom/{n}")
def get_bottom_n(n: int = 5, failure_type: Optional[str] = None):
    data = MACHINES

    if failure_type:
        data = [
            m
            for m in data
            if failure_type.lower() == m.get("predicted_failure_type", "").lower()
        ]

    sorted_data = sorted(data, key=lambda x: x["anomaly_probability"])
    return sorted_data[:n]


# -------------------------
# GET: Cari berdasarkan failure type / machine type
# -------------------------
@app.get("/machines/search")
def search(failure_type: Optional[str] = None, machine_type: Optional[str] = None):
    result = MACHINES

    if failure_type:
        result = [
            m
            for m in result
            if failure_type.lower() in m["predicted_failure_type"].lower()
        ]

    if machine_type:
        result = [
            m for m in result if m["Type"].lower() == machine_type.lower()
        ]

    return result


# -------------------------
# GET: Hitung jumlah mesin berdasarkan kriteria
# -------------------------
@app.get("/machines/search/count")
def search_count(failure_type: Optional[str] = None):
    result = MACHINES

    if failure_type:
        result = [
            m
            for m in result
            if failure_type.lower() in m.get("predicted_failure_type", "").lower()
        ]

    return {
        "status": "success",
        "count": len(result),
        "filter_applied": {"failure_type": failure_type},
    }


# -------------------------
# GET: Rekomendasi Maintenance (Filter + Sort Risk + Limit)
# -------------------------
@app.get("/machines/maintenance-candidates")
def get_maintenance_candidates(failure_type: Optional[str] = None, limit: int = 5):
    # Mulai dengan semua data
    candidates = MACHINES

    # Filter berdasarkan Failure Type (jika diminta user)
    if failure_type:
        candidates = [
            m
            for m in candidates
            if failure_type.lower() in m.get("predicted_failure_type", "").lower()
        ]

    # Filter: Hanya yang probability > 0 (biar ga bikin tiket buat mesin sehat)
    candidates = [m for m in candidates if m.get("anomaly_probability", 0) > 0]

    # Sorting: Urutkan dari Risiko TERTINGGI (Descending)
    sorted_candidates = sorted(
        candidates, key=lambda x: x.get("anomaly_probability", 0), reverse=True
    )

    # Limit: Ambil N teratas (Default 5 jika user tidak minta jumlah)
    final_result = sorted_candidates[:limit]

    return {"status": "success", "count": len(final_result), "candidates": final_result}


# ==========================================
#  Ringkasan kondisi global mesin
# ==========================================
@app.get("/stats/summary")
def get_factory_summary():
    total_machines = len(MACHINES)
    if total_machines == 0:
        return {"status": "empty", "message": "Belum ada data mesin."}

    # Hitung mesin sehat vs berisiko (Threshold > 50%)
    risky_machines = [m for m in MACHINES if m.get("anomaly_probability", 0) > 0.5]
    risky_count = len(risky_machines)
    healthy_count = total_machines - risky_count

    # Hitung Breakdown Kerusakan
    failure_counts = {}
    for m in MACHINES:
        ftype = m.get("predicted_failure_type", "No Failure")
        if ftype != "No Failure":
            failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

    # Cari kerusakan paling dominan
    top_failure = (
        max(failure_counts, key=failure_counts.get) if failure_counts else "Tidak ada"
    )

    return {
        "status": "success",
        "total_machines": total_machines,
        "health_status": {
            "healthy": healthy_count,
            "risky": risky_count,
            "risk_percentage": f"{(risky_count/total_machines)*100:.1f}%",
        },
        "most_common_failure": top_failure,
        "failure_breakdown": failure_counts,
    }


# -------------------------
# GET: Statistik Berdasarkan Tipe Mesin (L, M, H)
# -------------------------
@app.get("/stats/by-machine-type")
def get_stats_by_type():
    stats = {}

    # Inisialisasi pengelompokan
    for m in MACHINES:
        m_type = m.get(
            "Type", "Unknown"
        )  # Memastikan key JSON sesuai (misal: "Type" atau "machine_type")
        if m_type not in stats:
            stats[m_type] = {"total": 0, "failures": 0, "avg_risk": 0.0}

        stats[m_type]["total"] += 1
        stats[m_type]["avg_risk"] += m.get("anomaly_probability", 0)

        # Hitung jika ada kerusakan nyata (bukan "No Failure")
        if m.get("predicted_failure_type") != "No Failure":
            stats[m_type]["failures"] += 1

    # Rapikan hasil (hitung rata-rata)
    for m_type in stats:
        count = stats[m_type]["total"]
        if count > 0:
            stats[m_type]["avg_risk"] = round(stats[m_type]["avg_risk"] / count, 3)
            stats[m_type][
                "failure_rate"
            ] = f"{(stats[m_type]['failures'] / count)*100:.1f}%"

    return stats

# -------------------------
# GET: Rata-rata Sensor berdasarkan Failure Type
# -------------------------
@app.get("/stats/sensor-analysis")
def get_sensor_analysis(failure_type: str):
    # Filter mesin berdasarkan failure type target
    target_machines = [
        m
        for m in MACHINES
        if failure_type.lower() in m.get("predicted_failure_type", "").lower()
    ]

    if not target_machines:
        return {"message": f"No machines found with failure type: {failure_type}"}

    count = len(target_machines)

    # Menyesuaikan key di bawah dengan nama kolom di JSON/CSV
    # Contoh: "Air temperature", "Process temperature", "Rotational speed"
    avg_air_temp = sum(m.get("Air temperature [K]", 0) for m in target_machines) / count
    avg_proc_temp = (
        sum(m.get("Process temperature [K]", 0) for m in target_machines) / count
    )
    avg_rpm = sum(m.get("Rotational speed [rpm]", 0) for m in target_machines) / count

    return {
        "failure_type": failure_type,
        "analyzed_machines": count,
        "averages": {
            "air_temperature": round(avg_air_temp, 2),
            "process_temperature": round(avg_proc_temp, 2),
            "rotational_speed": round(avg_rpm, 2),
        },
    }


# -------------------------
# GET: Detail satu mesin
# -------------------------
@app.get("/machines/{product_id}")
def get_machine(product_id: str):
    for m in MACHINES:
        if m["Product ID"] == product_id:
            return m
    return {"error": "Machine not found"}
