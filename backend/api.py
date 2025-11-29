import os

# Change to project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print(f"Backend running from: {os.getcwd()}\n")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import pickle
import uuid
from datetime import datetime
import json
import warnings
import re

warnings.filterwarnings("ignore")

# Globals
MODELS = {}
PREPROCESSOR = None
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("backend/models", exist_ok=True)


# Same patterns and feature extractor as in scripts/train_models.py
sql_pattern = r"(union|select|drop|insert|--|'|or\s+1=1)"
xss_pattern = r"(<script|onerror|alert\()"


def extract_features(df, ip_counts=None, unique_uri=None):
    rows = []

    for _, r in df.iterrows():
        uri = str(r.get('uri', '') or '')
        payload = str(r.get('payload', '') or '')
        ua = str(r.get('user_agent', '') or '')
        method = str(r.get('method', '') or '')
        ip = r.get('ip', '')
        ts = r.get('timestamp')

        feature = {}

        feature['url_length'] = len(uri)
        feature['has_parameters'] = int('?' in uri)
        feature['param_count'] = uri.count('&') + 1 if '?' in uri else 0
        feature['path_depth'] = uri.count('/')
        feature['dot_count'] = uri.count('.')

        feature['payload_length'] = len(payload)
        feature['special_chars'] = sum(c in '<>"\'()=&;%' for c in payload)
        feature['encoded_chars'] = payload.count('%')
        feature['payload_sql'] = int(bool(re.search(sql_pattern, payload.lower())))
        feature['payload_xss'] = int(bool(re.search(xss_pattern, payload.lower())))

        feature['uri_sql'] = int(bool(re.search(sql_pattern, uri.lower())))
        feature['uri_xss'] = int(bool(re.search(xss_pattern, uri.lower())))

        feature['is_post'] = int(method.upper() == 'POST')

        feature['login_words'] = int(
            any(w in uri.lower() for w in ['login', 'signin', 'auth', 'admin', 'wp-login', 'account'])
        )
        feature['has_credentials'] = int(
            any(k in payload.lower() for k in ['username', 'user=', 'password', 'pass=', 'pwd=', 'login='])
        )

        feature['ua_length'] = len(ua)
        feature['is_bot'] = int(
            any(tok in ua.lower() for tok in [
                'bot', 'crawler', 'spider', 'sqlmap', 'nikto', 'curl', 'python', 'wget', 'libwww'
            ])
        )

        feature['requests_per_ip'] = int(ip_counts.get(ip, 1)) if ip_counts is not None else 1
        feature['unique_uri_per_ip'] = int(unique_uri.get(ip, 1)) if unique_uri is not None else 1

        feature['credential_ratio'] = feature['has_credentials'] / max(feature['requests_per_ip'], 1)

        feature['login_focus'] = int(
            feature['login_words'] == 1 and
            feature['is_post'] == 1 and
            feature['has_credentials'] == 1
        )

        total = feature['requests_per_ip']
        unique = feature['unique_uri_per_ip']

        feature['log_requests_per_ip'] = np.log1p(total)
        feature['log_unique_uri'] = np.log1p(unique)
        feature['uri_entropy'] = unique / max(total, 1)

        feature['ddos_like'] = int(
            feature['log_requests_per_ip'] > 4 and
            feature['uri_entropy'] < 0.2 and
            feature['login_words'] == 0
        )

        if payload:
            probs = [payload.count(c) / len(payload) for c in set(payload)]
            feature['payload_entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            feature['payload_entropy'] = 0.0

        feature['looks_human'] = int(
            feature['ua_length'] > 25 and
            feature['is_bot'] == 0 and
            feature['uri_entropy'] > 0.4 and
            feature['log_requests_per_ip'] < 3
        )

        if pd.notna(ts):
            feature['hour'] = ts.hour
            feature['weekday'] = ts.weekday()
        else:
            feature['hour'] = -1
            feature['weekday'] = -1

        rows.append(feature)

    return pd.DataFrame(rows)


def load_models_and_data():
    """Load Gradient Boosting model + preprocessor saved by scripts/train_models.py"""
    global MODELS, PREPROCESSOR

    print("\n" + "="*60)
    print("LOADING MODEL AND PREPROCESSOR")
    print("="*60 + "\n")

    try:
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}\n")

        # preprocessor.pkl
        preprocessor_paths = [
            'preprocessor.pkl',
            os.path.join(cwd, 'preprocessor.pkl'),
            os.path.abspath('preprocessor.pkl')
        ]
        preprocessor_loaded = False
        for p in preprocessor_paths:
            print(f"  Exists at {p}? {os.path.exists(p)}")
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        PREPROCESSOR = pickle.load(f)
                    print(f"✓ Preprocessor loaded from {p}")
                    preprocessor_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Error loading preprocessor from {p}: {e}")

        if not preprocessor_loaded:
            print("⚠️  Preprocessor not found. Run: python scripts/train_models.py")

        # model
        model_paths = [
            'models/gradient_boosting_threat_model.pkl',
            os.path.join('backend', 'models', 'gradient_boosting_threat_model.pkl')
        ]
        gb_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        model_data = pickle.load(f)
                    MODELS['gradient_boosting'] = model_data
                    print(f"✓ Gradient Boosting model loaded from {path}")
                    gb_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Error loading Gradient Boosting from {path}: {e}")

        if not gb_loaded:
            print("⚠️  Gradient Boosting model not found. Run: python scripts/train_models.py")

        if gb_loaded and preprocessor_loaded:
            print("\n✓ Model and preprocessor loaded successfully")
            print(f"  Label classes: {list(PREPROCESSOR['label_encoder'].classes_)}")
            print(f"  Num features: {len(PREPROCESSOR['feature_columns'])}")
        else:
            print("\n⚠️  Some required files are missing")

    except Exception as e:
        print(f"✗ Error loading model/preprocessor: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("APPLICATION STARTUP")
    print("="*60)
    load_models_and_data()
    print("="*60)
    print("APPLICATION READY")
    print("="*60 + "\n")
    yield
    print("\n" + "="*60)
    print("APPLICATION SHUTDOWN")
    print("="*60 + "\n")


app = FastAPI(
    title="Web Threat Prediction API (GB)",
    description="API using same Gradient Boosting pipeline as training.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    predictions: Optional[Dict] = None


class ThreatSummary(BaseModel):
    total_requests: int
    normal_traffic: int
    ddos_threats: int
    brute_force_threats: int
    xss_threats: int
    sql_injection_threats: int
    threat_percentage: float
    high_risk_ips: List[str]


class DetailedPrediction(BaseModel):
    timestamp: str
    ip: str
    uri: str
    predicted_threat: str
    confidence: float
    risk_score: float


class AnalysisResult(BaseModel):
    job_id: str
    summary: ThreatSummary
    detailed_predictions: List[DetailedPrediction]
    temporal_analysis: Dict
    recommendations: List[str]


@app.get("/")
async def root():
    return {
        "message": "Web Threat Prediction API (GB)",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload",
            "predict": "/api/predict/{job_id}",
            "status": "/api/status/{job_id}"
        }
    }


@app.get("/health")
async def health_check():
    models_loaded = 'gradient_boosting' in MODELS
    preprocessor_loaded = PREPROCESSOR is not None
    status = "healthy" if models_loaded and preprocessor_loaded else "degraded"
    return {
        "status": status,
        "models_loaded": models_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "available_models": list(MODELS.keys()),
        "details": {
            "gradient_boosting": models_loaded,
            "preprocessor": preprocessor_loaded,
            "label_classes": list(PREPROCESSOR['label_encoder'].classes_) if preprocessor_loaded else []
        }
    }


@app.post("/api/upload", response_model=PredictionResponse)
async def upload_log_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    allowed_extensions = ('.csv', '.tsv', '.txt', '.log')
    if not file.filename or not file.filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Please upload {', '.join(allowed_extensions)} file."
        )

    job_id = str(uuid.uuid4())
    original_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    try:
        contents = await file.read()
        with open(original_file_path, 'wb') as f:
            f.write(contents)

        background_tasks.add_task(process_uploaded_file, job_id, original_file_path)

        return PredictionResponse(
            job_id=job_id,
            status="processing",
            message=f"File uploaded successfully. Processing job {job_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )


async def process_uploaded_file(job_id: str, file_path: str):
    try:
        save_job_status(job_id, "processing", "Checking file format...")
        if file_path.endswith(('.log', '.txt')):
            csv_path = await convert_log_to_csv(job_id, file_path)
            await process_log_file(job_id, csv_path)
        else:
            await process_log_file(job_id, file_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        save_job_status(job_id, "error", str(e))


async def convert_log_to_csv(job_id: str, log_file_path: str) -> str:
    import csv
    import re

    csv_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_converted.csv")

    log_pattern = re.compile(
        r'(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\S+) (?P<uri>[^ ]+) HTTP/[0-9.]+" '
        r'(?P<status>\d+) \d+ '
        r'"(?P<refer>[^"]*)" '
        r'"(?P<user_agent>[^"]*)"'
    )

    def extract_payload(uri):
        if "?" in uri:
            return uri.split("?", 1)[1]
        return ""

    total_lines = 0
    parsed_lines = 0
    failed_lines = 0

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["ip", "timestamp", "method", "uri", "status", "payload", "refer", "user_agent"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                m = log_pattern.search(line)
                if m:
                    data = m.groupdict()
                    data["payload"] = extract_payload(data["uri"])
                    writer.writerow(data)
                    parsed_lines += 1
                else:
                    failed_lines += 1

    save_job_status(
        job_id,
        "processing",
        f"Converted log file: {parsed_lines}/{total_lines} lines parsed successfully"
    )

    return csv_file_path


async def process_log_file(job_id: str, file_path: str):
    results_file = os.path.join(RESULTS_DIR, f"{job_id}_results.json")

    try:
        save_job_status(job_id, "processing", "Loading log file...")

        df = None
        for delim in [',', '\t', ';']:
            try:
                df = pd.read_csv(file_path, delimiter=delim)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue

        if df is None or len(df.columns) == 1:
            raise Exception("Could not parse CSV file")

        if PREPROCESSOR is None or 'gradient_boosting' not in MODELS:
            raise Exception("Model or preprocessor not loaded")

        # Parse timestamp as in training
        df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            format='%d/%b/%Y:%H:%M:%S %z',
            errors='coerce'
        )

        save_job_status(job_id, "processing", "Extracting features...")

        ip_counts = PREPROCESSOR['train_ip_counts']
        unique_uri = PREPROCESSOR['train_unique_uri']
        feature_columns = PREPROCESSOR['feature_columns']
        scaler = PREPROCESSOR['scaler']
        label_encoder = PREPROCESSOR['label_encoder']

        X_feat = extract_features(df, ip_counts, unique_uri)

        for col in feature_columns:
            if col not in X_feat.columns:
                X_feat[col] = 0

        X_feat = X_feat[feature_columns]
        X_scaled = scaler.transform(X_feat.values)

        save_job_status(job_id, "processing", "Making predictions...")

        model = MODELS['gradient_boosting']['model']
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        results = create_analysis_results(df, preds, probs, job_id, label_encoder)
        with open(results_file, 'w') as f:
            json.dump(results, f)

        save_job_status(job_id, "completed", "Analysis complete")
    except Exception as e:
        import traceback
        traceback.print_exc()
        save_job_status(job_id, "error", str(e))


def save_job_status(job_id: str, status: str, message: str):
    status_file = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
    data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    with open(status_file, 'w') as f:
        json.dump(data, f)


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    status_file = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Job not found")
    with open(status_file, 'r') as f:
        return json.load(f)


@app.get("/api/predict/{job_id}", response_model=AnalysisResult)
async def get_predictions(job_id: str):
    results_file = os.path.join(RESULTS_DIR, f"{job_id}_results.json")
    if not os.path.exists(results_file):
        status_file = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
            if status['status'] == 'processing':
                raise HTTPException(status_code=202, detail="Job still processing.")
        raise HTTPException(status_code=404, detail="Results not found")
    with open(results_file, 'r') as f:
        return json.load(f)


def create_analysis_results(df, predictions, probabilities, job_id, label_encoder):
    class_names = label_encoder.classes_

    threat_labels = list(class_names)  # direct mapping by encoded index

    unique, counts = np.unique(predictions, return_counts=True)
    idx_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    total = len(df)
    normal_idx = np.where(class_names == 'normal')[0][0] if 'normal' in class_names else None

    summary = {
        "total_requests": int(total),
        "normal_traffic": int(idx_counts.get(normal_idx, 0)) if normal_idx is not None else 0,
        "ddos_threats": int(idx_counts.get(np.where(class_names == 'ddos')[0][0], 0)) if 'ddos' in class_names else 0,
        "brute_force_threats": int(idx_counts.get(np.where(class_names == 'brute_force')[0][0], 0)) if 'brute_force' in class_names else 0,
        "xss_threats": int(idx_counts.get(np.where(class_names == 'xss')[0][0], 0)) if 'xss' in class_names else 0,
        "sql_injection_threats": int(idx_counts.get(np.where(class_names == 'sql_injection')[0][0], 0)) if 'sql_injection' in class_names else 0,
        "threat_percentage": float((total - idx_counts.get(normal_idx, 0)) / total * 100) if total > 0 and normal_idx is not None else 0.0,
        "high_risk_ips": []
    }

    if 'ip' in df.columns and normal_idx is not None:
        threat_mask = predictions != normal_idx
        if threat_mask.sum() > 0:
            high_risk_ips = df.loc[threat_mask, 'ip'].value_counts().head(5)
            summary['high_risk_ips'] = [str(ip) for ip in high_risk_ips.index.tolist()]

    detailed = []
    max_rows = min(100, len(df))
    for i in range(max_rows):
        conf = float(probabilities[i].max())
        cls_idx = int(predictions[i])
        label = threat_labels[cls_idx] if cls_idx < len(threat_labels) else 'unknown'
        detailed.append({
            "timestamp": str(df.iloc[i].get('timestamp', '')),
            "ip": str(df.iloc[i].get('ip', '')),
            "uri": str(df.iloc[i].get('uri', ''))[:100],
            "predicted_threat": label,
            "confidence": conf,
            "risk_score": conf if label != 'normal' else 0.0
        })

    temporal = {
        "total_time_windows": 1,
        "peak_threat_period": "N/A",
        "threat_trend": "stable"
    }

    recommendations = generate_recommendations(summary)

    return {
        "job_id": str(job_id),
        "summary": summary,
        "detailed_predictions": detailed,
        "temporal_analysis": temporal,
        "recommendations": recommendations
    }


def generate_recommendations(summary):
    recs = []

    if summary['ddos_threats'] > 0:
        recs.append("⚠ DDoS Attack Detected: Implement rate limiting and consider using CDN/DDoS protection")
    if summary['brute_force_threats'] > 0:
        recs.append("⚠ Brute Force Detected: Enable account lockout policies and implement CAPTCHA")
    if summary['xss_threats'] > 0:
        recs.append("⚠ XSS Attempts Detected: Implement input validation and CSP")
    if summary['sql_injection_threats'] > 0:
        recs.append("⚠ SQL Injection Detected: Use parameterized queries and input sanitization")
    if summary['threat_percentage'] > 50:
        recs.append("🚨 Critical: Over 50% malicious traffic detected. Consider blocking suspicious IPs")
    if len(summary['high_risk_ips']) > 0:
        recs.append(f"🔒 Monitor/Block these IPs: {', '.join(summary['high_risk_ips'][:3])}")
    if not recs:
        recs.append("✓ No immediate threats detected. Continue monitoring.")

    return recs


if __name__ == "__main__":
    import uvicorn
    print("Starting Web Threat Prediction API (GB)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
