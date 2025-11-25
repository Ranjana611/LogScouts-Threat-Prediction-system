import os

# Change to project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print(f"Backend running from: {os.getcwd()}\n")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import pickle
import uuid
from datetime import datetime
import json

# Global variables for models and processors
MODELS = {}
PREPROCESSOR = None
FEATURE_EXTRACTOR = None
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("backend/models", exist_ok=True)


def load_models_and_data():
    """Load trained models and preprocessors"""
    global MODELS, PREPROCESSOR, FEATURE_EXTRACTOR

    print("\n" + "="*60)
    print("LOADING MODELS AND PREPROCESSORS")
    print("="*60 + "\n")

    try:
        # Get current working directory
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}\n")

        # Load preprocessor - try multiple locations
        preprocessor_paths = [
            'preprocessor.pkl',
            os.path.join(cwd, 'preprocessor.pkl'),
            os.path.abspath('preprocessor.pkl')
        ]

        preprocessor_loaded = False
        for preprocessor_path in preprocessor_paths:
            if os.path.exists(preprocessor_path):
                try:
                    with open(preprocessor_path, 'rb') as f:
                        PREPROCESSOR = pickle.load(f)
                    print(f"✓ Preprocessor loaded from {preprocessor_path}")
                    preprocessor_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Error loading from {preprocessor_path}: {e}")

        if not preprocessor_loaded:
            print(f"⚠️  Preprocessor not found in any location:")
            for path in preprocessor_paths:
                print(f"   - {path}")
            print("   Run: python scripts/preprocess_data.py")

        # Try both locations for models
        model_configs = [
            ('random_forest', ['models/random_forest_threat_model.pkl',
             'backend/models/random_forest_threat_model.pkl']),
            ('gradient_boosting', ['models/gradient_boosting_threat_model.pkl',
             'backend/models/gradient_boosting_threat_model.pkl']),
        ]

        for model_name, paths in model_configs:
            loaded = False
            for path in paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            model_data = pickle.load(f)
                            MODELS[model_name] = model_data
                        print(f"✓ {model_name} model loaded from {path}")
                        loaded = True
                        break
                    except Exception as e:
                        print(
                            f"⚠️  Error loading {model_name} from {path}: {e}")

            if not loaded:
                print(f"⚠️  {model_name} model not found in any location")

        print()

        if len(MODELS) > 0 and preprocessor_loaded:
            print("✓ All models loaded successfully")
            print(f"  Available models: {list(MODELS.keys())}")
            print(f"  Preprocessor loaded: {PREPROCESSOR is not None}\n")
        else:
            print(f"⚠️  Some required files are missing")
            if not preprocessor_loaded:
                print("   - Preprocessor not loaded")
            if len(MODELS) == 0:
                print("   - No models loaded\n")

    except Exception as e:
        print(f"✗ Error loading models: {e}\n")
        import traceback
        traceback.print_exc()


# Lifespan context manager (modern approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for startup and shutdown events
    """
    # Startup
    print("\n" + "="*60)
    print("APPLICATION STARTUP")
    print("="*60)
    load_models_and_data()
    print("="*60)
    print("APPLICATION READY")
    print("="*60 + "\n")

    yield

    # Shutdown
    print("\n" + "="*60)
    print("APPLICATION SHUTDOWN")
    print("="*60 + "\n")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Web Threat Prediction API",
    description="ML-powered API for predicting web server threats",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response


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
    """Root endpoint with API information"""
    return {
        "message": "Web Threat Prediction API",
        "version": "1.0.0",
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
    """Health check endpoint"""
    models_loaded = len(MODELS) > 0
    preprocessor_loaded = PREPROCESSOR is not None

    health_status = "healthy" if (
        models_loaded and preprocessor_loaded) else "degraded"

    print(f"\nHealth Check:")
    print(f"  PREPROCESSOR is None: {PREPROCESSOR is None}")
    print(f"  MODELS: {list(MODELS.keys())}")
    print(f"  preprocessor.pkl exists: {os.path.exists('preprocessor.pkl')}")
    print()

    return {
        "status": health_status,
        "models_loaded": models_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "available_models": list(MODELS.keys()),
        "details": {
            "random_forest": os.path.exists('models/random_forest_threat_model.pkl') or os.path.exists('backend/models/random_forest_threat_model.pkl'),
            "gradient_boosting": os.path.exists('models/gradient_boosting_threat_model.pkl') or os.path.exists('backend/models/gradient_boosting_threat_model.pkl'),
            "preprocessor": os.path.exists('preprocessor.pkl'),
            "preprocessor_object": PREPROCESSOR is not None,
            "preprocessed_data": os.path.exists('data/processed/X_train.npy')
        }
    }

@app.post("/api/upload", response_model=PredictionResponse)
async def upload_log_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    print(f"📥 Upload attempt - Filename: {file.filename}")
    print(f"📥 Content type: {file.content_type}")
    
    # Validate file
    allowed_extensions = ('.csv', '.tsv', '.txt', '.log')
    if not file.filename or not file.filename.endswith(allowed_extensions):
        print(f"❌ Invalid file format: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Please upload {', '.join(allowed_extensions)} file."
        )
    
    print(f"✅ File accepted: {file.filename}")
    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    original_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    try:
        contents = await file.read()
        with open(original_file_path, 'wb') as f:
            f.write(contents)

        # Add processing task to background
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
    """Background task to process uploaded file (convert if needed) and make predictions"""
    
    try:
        save_job_status(job_id, "processing", "Checking file format...")
        
        # Check if file is a log file that needs conversion
        if file_path.endswith(('.log', '.txt')):
            save_job_status(job_id, "processing", "Converting log file to CSV...")
            csv_file_path = await convert_log_to_csv(job_id, file_path)
            # Process the converted CSV file
            await process_log_file(job_id, csv_file_path)
        else:
            # Process directly as CSV/TSV
            await process_log_file(job_id, file_path)
            
    except Exception as e:
        print(f"✗ Error processing uploaded file {job_id}: {e}")
        import traceback
        traceback.print_exc()
        save_job_status(job_id, "error", str(e))


async def convert_log_to_csv(job_id: str, log_file_path: str) -> str:
    """
    Convert Apache log file to CSV format
    """
    import re
    import csv
    
    # Define output CSV path
    csv_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_converted.csv")
    
    # Regex pattern to parse the log line
    log_pattern = re.compile(
        r'(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\S+) (?P<uri>[^ ]+) HTTP/[0-9.]+" '
        r'(?P<status>\d+) \d+ '
        r'"(?P<refer>[^"]*)" '
        r'"(?P<user_agent>[^"]*)"'
    )
    
    # Function to extract payload if present (query string)
    def extract_payload(uri):
        if "?" in uri:
            return uri.split("?", 1)[1]
        return ""
    
    # Counters for statistics
    total_lines = 0
    parsed_lines = 0
    failed_lines = 0
    
    # Open CSV and write headers
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["ip", "timestamp", "method", "uri", "status", "payload", "refer", "user_agent"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                match = log_pattern.search(line)
                if match:
                    data = match.groupdict()
                    data["payload"] = extract_payload(data["uri"])
                    writer.writerow(data)
                    parsed_lines += 1
                else:
                    print(f"❌ No match for line {total_lines}: {line[:100]}...")
                    failed_lines += 1
    
    # Log conversion statistics
    print(f"✓ Log conversion complete for job {job_id}")
    print(f"  Total lines: {total_lines}")
    print(f"  Successfully parsed: {parsed_lines}")
    print(f"  Failed to parse: {failed_lines}")
    print(f"  Success rate: {(parsed_lines/total_lines)*100:.2f}%")
    
    save_job_status(job_id, "processing", 
                   f"Converted log file: {parsed_lines}/{total_lines} lines parsed successfully")
    
    return csv_file_path


async def process_log_file(job_id: str, file_path: str):
    """Background task to process log file and make predictions"""
    # ... rest of your existing process_log_file function remains exactly the same ...
    results_file = os.path.join(RESULTS_DIR, f"{job_id}_results.json")

    try:
        save_job_status(job_id, "processing", "Loading log file...")

        # Load file with auto-detection
        df = None
        for delim in [',', '\t', ';']:
            try:
                df = pd.read_csv(file_path, delimiter=delim)
                if len(df.columns) > 1:
                    print(f"✓ Loaded with delimiter: {repr(delim)}")
                    break
            except:
                continue

        if df is None or len(df.columns) == 1:
            raise Exception("Could not parse CSV file")

        print(f"Loaded {len(df)} records from {file_path}")
        save_job_status(job_id, "processing",
                        f"Loaded {len(df)} log entries...")

        # Check models
        if not MODELS or 'random_forest' not in MODELS:
            raise Exception("No ML models available")

        if PREPROCESSOR is None:
            raise Exception("Preprocessor not available")

        # Extract features
        save_job_status(job_id, "processing", "Extracting features...")

        from feature_extractor import ThreatFeatureExtractor

        feature_extractor = ThreatFeatureExtractor()
        feature_matrix = feature_extractor.create_feature_matrix(df)

        print(f"✓ Features extracted: {feature_matrix.shape}")

        # Get feature columns
        feature_cols = PREPROCESSOR['feature_columns']

        print(f"Feature columns needed: {len(feature_cols)}")

        # Check if all required features exist
        missing_features = [
            f for f in feature_cols if f not in feature_matrix.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            for feat in missing_features:
                feature_matrix[feat] = 0

        # ============================================================
        # IMPORTANT: PREPROCESS FEATURES BEFORE PREDICTION
        # ============================================================

        save_job_status(job_id, "processing", "Preprocessing features...")

        # Get the raw features in correct order
        X_raw = feature_matrix[feature_cols]

        print(f"Raw features shape: {X_raw.shape}")
        print(f"Raw features sample:\n{X_raw.head(2)}")

        # Handle categorical features (encode them)
        categorical_mappings = PREPROCESSOR.get('categorical_mappings', {})

        for col in feature_cols:
            if col in categorical_mappings:
                # Apply the mapping used during training
                X_raw[col] = X_raw[col].map(
                    categorical_mappings[col]).fillna(-1).astype(int)
            elif X_raw[col].dtype == 'object' or X_raw[col].dtype == 'bool':
                # Convert boolean to int
                if X_raw[col].dtype == 'bool':
                    X_raw[col] = X_raw[col].astype(int)
                else:
                    # For any unmapped strings, create a simple encoding
                    unique_vals = X_raw[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    X_raw[col] = X_raw[col].map(mapping).fillna(-1).astype(int)

        # Replace inf values
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with median or 0
        X_raw = X_raw.fillna(0)

        # Convert to numpy array
        X_prepared = X_raw.values

        print(f"Prepared features shape: {X_prepared.shape}")
        print(f"Features dtype: {X_prepared.dtype}")

        # Scale features using the saved scaler
        scaler = PREPROCESSOR.get('scaler')
        if scaler is not None:
            print("Scaling features...")
            X_scaled = scaler.transform(X_prepared)
        else:
            print("⚠️  No scaler found, using unscaled features")
            X_scaled = X_prepared

        print(f"Scaled features shape: {X_scaled.shape}")

        # Verify all values are numeric
        if not np.issubdtype(X_scaled.dtype, np.number):
            raise Exception(
                f"Features contain non-numeric values: {X_scaled.dtype}")

        # Make predictions
        save_job_status(job_id, "processing", "Making predictions...")

        model = MODELS['random_forest']['model']

        print(f"Making predictions on {X_scaled.shape[0]} samples...")
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        print(f"✓ Predictions complete")
        print(f"  Unique predictions: {np.unique(predictions)}")

        # Create and save results
        results = create_analysis_results(
            df, predictions, probabilities, job_id)

        with open(results_file, 'w') as f:
            json.dump(results, f)

        print(f"✓ Predictions saved for job {job_id}")
        save_job_status(job_id, "completed", "Analysis complete")

    except Exception as e:
        print(f"✗ Error processing job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        save_job_status(job_id, "error", str(e))
def create_fallback_features(df, feature_columns):
    # Create features using basic methods if feature extractor is not available
    features_list = []

    for idx, row in df.iterrows():
        features = {}

        # Basic features from available columns
        features['url_length'] = len(str(row.get('uri', '')))
        features['param_count'] = str(row.get('uri', '')).count(
            '&') + 1 if '?' in str(row.get('uri', '')) else 0
        features['path_depth'] = str(row.get('uri', '')).count('/')

        features['is_post'] = 1 if str(
            row.get('method', 'GET')) == 'POST' else 0
        features['is_get'] = 1 if str(row.get('method', 'GET')) == 'GET' else 0

        status = int(row.get('status', 200))
        features['status_2xx'] = 1 if 200 <= status < 300 else 0
        features['status_3xx'] = 1 if 300 <= status < 400 else 0
        features['status_4xx'] = 1 if 400 <= status < 500 else 0
        features['status_5xx'] = 1 if 500 <= status < 600 else 0

        features['ua_length'] = len(str(row.get('user_agent', '')))

        features_list.append(features)

    feature_df = pd.DataFrame(features_list)

    # Fill missing columns with 0
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0

    # Select only required columns in correct order
    X = feature_df[feature_columns].values

    return X


def save_job_status(job_id: str, status: str, message: str):
    """Save job status to file"""
    status_file = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
    status_data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

    with open(status_file, 'w') as f:
        json.dump(status_data, f)


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status of a job"""
    status_file = os.path.join(RESULTS_DIR, f"{job_id}_status.json")

    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Job not found")

    with open(status_file, 'r') as f:
        status_data = json.load(f)

    return status_data


@app.get("/api/predict/{job_id}", response_model=AnalysisResult)
async def get_predictions(job_id: str):
    """Get prediction results for a completed job"""
    results_file = os.path.join(RESULTS_DIR, f"{job_id}_results.json")

    if not os.path.exists(results_file):
        # Check if job is still processing
        status_file = os.path.join(RESULTS_DIR, f"{job_id}_status.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)

            if status['status'] == 'processing':
                raise HTTPException(
                    status_code=202,
                    detail="Job still processing. Please try again later."
                )

        raise HTTPException(status_code=404, detail="Results not found")

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def create_analysis_results(df, predictions, probabilities, job_id):
    """Create comprehensive analysis results"""

    # Threat label mapping
    threat_labels = ['normal', 'ddos', 'brute_force', 'xss', 'sql_injection']

    # Count threats
    unique, counts = np.unique(predictions, return_counts=True)
    threat_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    # Create summary
    total = len(df)
    summary = {
        "total_requests": int(total),
        "normal_traffic": int(threat_counts.get(0, 0)),
        "ddos_threats": int(threat_counts.get(1, 0)),
        "brute_force_threats": int(threat_counts.get(2, 0)),
        "xss_threats": int(threat_counts.get(3, 0)),
        "sql_injection_threats": int(threat_counts.get(4, 0)),
        "threat_percentage": float((total - threat_counts.get(0, 0)) / total * 100) if total > 0 else 0,
        "high_risk_ips": []
    }

    # Find high-risk IPs
    if 'ip' in df.columns:
        threat_mask = predictions != 0
        if threat_mask.sum() > 0:
            high_risk_ips = df.loc[threat_mask, 'ip'].value_counts().head(5)
            summary['high_risk_ips'] = [str(ip)
                                        for ip in high_risk_ips.index.tolist()]

    # Detailed predictions
    detailed = []
    for idx in range(min(100, len(df))):
        confidence = float(probabilities[idx].max())
        pred_class = int(predictions[idx])

        detailed.append({
            "timestamp": str(df.iloc[idx].get('timestamp', '')),
            "ip": str(df.iloc[idx].get('ip', '')),
            "uri": str(df.iloc[idx].get('uri', ''))[:100],
            "predicted_threat": threat_labels[pred_class] if pred_class < len(threat_labels) else 'unknown',
            "confidence": confidence,
            "risk_score": confidence if pred_class != 0 else 0.0
        })

    # Temporal analysis
    temporal = {
        "total_time_windows": int(1),
        "peak_threat_period": "N/A",
        "threat_trend": "stable"
    }

    # Recommendations
    recommendations = generate_recommendations(summary)

    return {
        "job_id": str(job_id),
        "summary": summary,
        "detailed_predictions": detailed,
        "temporal_analysis": temporal,
        "recommendations": recommendations
    }


def generate_recommendations(summary):
    """Generate security recommendations based on detected threats"""
    recommendations = []

    if summary['ddos_threats'] > 0:
        recommendations.append(
            "⚠ DDoS Attack Detected: Implement rate limiting and consider using CDN/DDoS protection"
        )

    if summary['brute_force_threats'] > 0:
        recommendations.append(
            "⚠ Brute Force Detected: Enable account lockout policies and implement CAPTCHA"
        )

    if summary['xss_threats'] > 0:
        recommendations.append(
            "⚠ XSS Attempts Detected: Implement input validation and Content Security Policy (CSP)"
        )

    if summary['sql_injection_threats'] > 0:
        recommendations.append(
            "⚠ SQL Injection Detected: Use parameterized queries and input sanitization"
        )

    if summary['threat_percentage'] > 50:
        recommendations.append(
            "🚨 Critical: Over 50% malicious traffic detected. Consider blocking suspicious IPs"
        )

    if len(summary['high_risk_ips']) > 0:
        recommendations.append(
            f"🔒 Monitor/Block these IPs: {', '.join(summary['high_risk_ips'][:3])}"
        )

    if not recommendations:
        recommendations.append(
            "✓ No immediate threats detected. Continue monitoring.")

    return recommendations


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job results and uploaded files"""
    deleted_files = []

    # Delete all related files
    import glob
    patterns = [
        os.path.join(UPLOAD_DIR, f"{job_id}_*"),
        os.path.join(RESULTS_DIR, f"{job_id}_*")
    ]

    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except:
                pass

    return {
        "message": f"Deleted {len(deleted_files)} files for job {job_id}",
        "deleted_files": deleted_files
    }


@app.get("/api/models")
async def get_available_models():
    """Get information about loaded models"""
    return {
        "loaded_models": list(MODELS.keys()),
        "model_info": {
            name: {
                "type": data.get('model_type', 'unknown'),
                "training_history": data.get('training_history', {})
            }
            for name, data in MODELS.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Web Threat Prediction API...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
