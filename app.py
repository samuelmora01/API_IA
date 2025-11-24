from flask import Flask, request, jsonify
import flask
import joblib
import numpy as np
import os
import sys
import time
from datetime import datetime
import sklearn
import json
from flask import send_file
import csv

app = Flask(__name__)

model = joblib.load("./models/modelo_regresion_logistica.pkl")
scaler = joblib.load('./models/scaler.pkl')

APP_VERSION = "v1"
APP_START_TIME = time.time()

STORAGE_DIR = "./storage"
STORAGE_PKL = os.path.join(STORAGE_DIR, "data.pkl")
STORAGE_JSON = os.path.join(STORAGE_DIR, "data.json")
STORAGE_CSV = os.path.join(STORAGE_DIR, "data.csv")
os.makedirs(STORAGE_DIR, exist_ok=True)

def _num_expected_features():
    # Try to infer expected feature count from scaler or model if available
    for obj in (scaler, model):
        n = getattr(obj, "n_features_in_", None)
        if isinstance(n, (int, np.integer)) and n > 0:
            return int(n)
    return None

def _file_info(path):
    try:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else None
        mtime = os.path.getmtime(path) if exists else None
        mtime_iso = datetime.fromtimestamp(mtime).isoformat() if mtime else None
        return {"exists": exists, "size_bytes": size, "modified": mtime_iso}
    except Exception as exc:
        return {"exists": False, "error": str(exc)}

def _versions():
    return {
        "python": sys.version.split(" ")[0],
        "flask": getattr(flask, "__version__", None),
        "sklearn": getattr(sklearn, "__version__", None),
        "numpy": getattr(np, "__version__", None),
        "joblib": getattr(joblib, "__version__", None),
    }

def _model_metadata():
    # Try to read external metadata if exists, fallback to embedded defaults
    default = {
        "modelo": "Regresión Logística",
        "random_state": 42,
        "max_iter": 1000,
        "domain": {
            "condition": "Hepatitis",
            "target_labels": {
                "1": "Vive",
                "2": "Muere"
            }
        },
        "metricas_train": {
            "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0
        },
        "metricas_test": {
            "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0
        },
        "n_features": 21,
        "features": [
            "Age","Sex","Estado_Civil","Ciudad","Steroid","Antivirals","Fatigue",
            "Malaise","Anorexia","Liver_Big","Liver_Firm","Spleen_Palpable","Spiders",
            "Ascites","Varices","Bilirubin","Alk_Phosphate","Sgot","Albumin","Protime",
            "Histology"
        ],
        "feature_schema": [
            {"name":"Age","type":"number","unit":"years","description":"Edad del paciente","suggested_range":[0,120]},
            {"name":"Sex","type":"categorical","values":[0,1],"labels":{"0":"Female","1":"Male"},"description":"Sexo biológico"},
            {"name":"Estado_Civil","type":"categorical","description":"Estado civil codificado (numérico/categórico)"},
            {"name":"Ciudad","type":"categorical","description":"Ciudad de atención codificada"},
            {"name":"Steroid","type":"binary","values":[0,1],"description":"Uso de esteroides"},
            {"name":"Antivirals","type":"binary","values":[0,1],"description":"Uso de antivirales"},
            {"name":"Fatigue","type":"binary","values":[0,1],"description":"Fatiga"},
            {"name":"Malaise","type":"binary","values":[0,1],"description":"Malestar general"},
            {"name":"Anorexia","type":"binary","values":[0,1],"description":"Anorexia"},
            {"name":"Liver_Big","type":"binary","values":[0,1],"description":"Hígado aumentado"},
            {"name":"Liver_Firm","type":"binary","values":[0,1],"description":"Hígado firme"},
            {"name":"Spleen_Palpable","type":"binary","values":[0,1],"description":"Bazo palpable"},
            {"name":"Spiders","type":"binary","values":[0,1],"description":"Arañas vasculares"},
            {"name":"Ascites","type":"binary","values":[0,1],"description":"Ascitis"},
            {"name":"Varices","type":"binary","values":[0,1],"description":"Várices"},
            {"name":"Bilirubin","type":"number","unit":"mg/dL","description":"Bilirrubina sérica","typical_range":[0.1,8.0]},
            {"name":"Alk_Phosphate","type":"number","unit":"U/L","description":"Fosfatasa alcalina","typical_range":[20,140]},
            {"name":"Sgot","type":"number","unit":"U/L","description":"AST/SGOT","typical_range":[10,40]},
            {"name":"Albumin","type":"number","unit":"g/dL","description":"Albúmina sérica","typical_range":[3.5,5.5]},
            {"name":"Protime","type":"number","unit":"seconds","description":"Tiempo de protrombina","typical_range":[9,14]},
            {"name":"Histology","type":"binary","values":[0,1],"description":"Hallazgos histológicos"}
        ],
        "example_payload": {
            "features": [45,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1.2,100,35,4.2,12,0]
        }
    }
    try:
        meta_path = "./models/model_info.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
    except Exception:
        pass
    return default

def _read_storage_list():
    # Returns list of records stored
    try:
        if os.path.exists(STORAGE_PKL):
            data = joblib.load(STORAGE_PKL)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def _write_storage_list(records):
    # Persist to PKL, JSON, and CSV
    try:
        joblib.dump(records, STORAGE_PKL)
    except Exception:
        pass
    try:
        with open(STORAGE_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        # Determine CSV header by union of keys across records
        header = set()
        for r in records:
            if isinstance(r, dict):
                header.update(r.keys())
        header = list(sorted(header))
        with open(STORAGE_CSV, "w", newline="", encoding="utf-8") as f:
            if not header:
                f.write("")  # empty
            else:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for r in records:
                    if isinstance(r, dict):
                        writer.writerow({k: r.get(k) for k in header})
    except Exception:
        pass

def _normalize_item(obj):
    # Accept either {"features":[...]} or dict keyed by feature names
    meta = _model_metadata()
    feature_names = meta.get("features") or []
    timestamp = datetime.utcnow().isoformat() + "Z"
    if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], list):
        values = obj["features"]
        if feature_names and len(values) != len(feature_names):
            raise ValueError(f"Invalid number of features. expected={len(feature_names)} received={len(values)}")
        record = {feature_names[i]: values[i] for i in range(len(values))} if feature_names else {str(i): v for i, v in enumerate(values)}
        record["_created_at"] = timestamp
        return record
    if isinstance(obj, dict):
        record = dict(obj)
        record["_created_at"] = timestamp
        return record
    raise ValueError("Unsupported payload item. Provide an object with 'features' list or feature fields.")

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "name": "Hepatitis API",
        "version": APP_VERSION,
        "status": "ok",
        "uptime_seconds": int(time.time() - APP_START_TIME),
        "endpoints": {
            "GET /health": "Health check",
            "GET /model": "Model metadata and feature names",
            "GET /data": "List stored records",
            "GET /data/file": "Download stored data file (format=json|csv|pkl)",
            "POST /data": "Append new record(s) to storage",
            "POST /predict": "Predict outcome given 'features' as numeric list"
        },
        "model": {
            "expected_features": _num_expected_features(),
            "classes": getattr(model, "classes_", []).tolist() if hasattr(model, "classes_") else None,
            "info": _model_metadata(),
            "files": {
                "model_pkl": _file_info("./models/modelo_regresion_logistica.pkl"),
                "scaler_pkl": _file_info("./models/scaler.pkl")
            }
        },
        "environment": _versions()
    })


@app.route('/health', methods=['GET'])
def health():
    try:
        # minimal no-op to ensure model is loaded
        _ = getattr(model, "classes_", None)
        payload = {
            "status": "ok",
            "ready": True,
            "uptime_seconds": int(time.time() - APP_START_TIME),
            "expected_features": _num_expected_features(),
            "classes": getattr(model, "classes_", []).tolist() if hasattr(model, "classes_") else None,
            "model_info": _model_metadata(),
            "files": {
                "model_pkl": _file_info("./models/modelo_regresion_logistica.pkl"),
                "scaler_pkl": _file_info("./models/scaler.pkl")
            },
            "versions": _versions()
        }
        return jsonify(payload), 200
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500


@app.route('/data', methods=['GET'])
def list_data():
    records = _read_storage_list()
    return jsonify({
        "count": len(records),
        "items": records,
        "files": {
            "pkl": _file_info(STORAGE_PKL),
            "json": _file_info(STORAGE_JSON),
            "csv": _file_info(STORAGE_CSV)
        }
    }), 200


@app.route('/data', methods=['POST'])
def append_data():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400
    items = None
    if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
        items = payload["items"]
    else:
        # accept single object as item
        items = [payload]
    try:
        normalized = [_normalize_item(it) for it in items]
    except Exception as exc:
        return jsonify({"error": "Invalid items", "detail": str(exc)}), 400
    existing = _read_storage_list()
    existing.extend(normalized)
    _write_storage_list(existing)
    return jsonify({"status": "ok", "added": len(normalized), "total": len(existing)}), 201


@app.route('/data/file', methods=['GET'])
def download_data_file():
    fmt = request.args.get("format", "json").lower()
    as_attachment = request.args.get("download", "1") != "0"
    path = None
    mimetype = None
    if fmt == "json":
        path = STORAGE_JSON
        mimetype = "application/json"
    elif fmt == "csv":
        path = STORAGE_CSV
        mimetype = "text/csv"
    elif fmt == "pkl":
        path = STORAGE_PKL
        mimetype = "application/octet-stream"
    else:
        return jsonify({"error": "Unsupported format", "supported": ["json","csv","pkl"]}), 400
    # Ensure files exist even if empty
    if not os.path.exists(path):
        _write_storage_list(_read_storage_list())
    return send_file(path, mimetype=mimetype, as_attachment=as_attachment)


@app.route('/model', methods=['GET'])
def model_info():
    # Dedicated endpoint to fetch model metadata and helpful details
    try:
        payload = {
            "info": _model_metadata(),
            "expected_features": _num_expected_features(),
            "classes": getattr(model, "classes_", []).tolist() if hasattr(model, "classes_") else None,
            "files": {
                "model_pkl": _file_info("./models/modelo_regresion_logistica.pkl"),
                "scaler_pkl": _file_info("./models/scaler.pkl")
            },
            "versions": _versions()
        }
        return jsonify(payload), 200
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    if "features" not in data:
        return jsonify({"error": "Missing 'features' field"}), 400

    features_raw = data["features"]
    if not isinstance(features_raw, list):
        return jsonify({"error": "'features' must be a list of numbers"}), 400

    try:
        features = np.array(features_raw, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({"error": "'features' must contain only numeric values"}), 400

    expected = _num_expected_features()
    if expected is not None and features.shape[1] != expected:
        return jsonify({
            "error": "Invalid number of features",
            "expected": expected,
            "received": int(features.shape[1])
        }), 400

    try:
        features_scaled = scaler.transform(features)
    except Exception as exc:
        return jsonify({"error": "Failed to scale features", "detail": str(exc)}), 500

    try:
        # Predicción (clase 1 o 2)
        prediction = model.predict(features_scaled)[0]

        probas = model.predict_proba(features_scaled)[0]
        classes = model.classes_  # e.g., [1, 2]

        proba_dict = {classes[i]: float(probas[i]) for i in range(len(classes))}

        # Sacar probabilidades según el significado real
        prob_vive = float(proba_dict.get(1, 0.0) * 100)   # Clase 1 = Vive
        prob_muere = float(proba_dict.get(2, 0.0) * 100)  # Clase 2 = Muere

        # Estado final
        estado = "Vive" if prediction == 1 else "Muere"

        return jsonify({
            "estado": estado,
            "prob_vive": prob_vive,
            "prob_muere": prob_muere
        })
    except Exception as exc:
        return jsonify({"error": "Model inference failed", "detail": str(exc)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
