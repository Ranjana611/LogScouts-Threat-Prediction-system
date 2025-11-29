"""
Model Training Script (Gradient Boosting, Colab-style features)
"""


import json
import sys
import os
import re 
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

backend_path = os.path.join(project_root, 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

os.chdir(project_root)

print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}\n")

# Import model wrapper
try:
    from model_trainer import ThreatPredictionModel
    from preprocessor import DataPreprocessor  # just for label_encoder reuse if needed
    print("✓ All modules imported successfully\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


# Colab-style feature extraction (same as in your notebook)
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


print("="*60)
print("MODEL TRAINING SCRIPT (Gradient Boosting, Colab-like)")
print("="*60 + "\n")

# Step 1: Load raw CSV (same as Colab)
data_file = 'data/raw/server_logs.csv'
if not os.path.exists(data_file):
    print(f"✗ Error: Data file not found: {data_file}")
    sys.exit(1)

print(f"Step 1: Loading dataset from {data_file}...")
df = pd.read_csv(data_file)
print(f"✓ Dataset loaded: {df.shape}\n")

# Parse timestamps exactly as in Colab
df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='%d/%b/%Y:%H:%M:%S %z',
    errors='coerce'
)

# IP stats on full dataset
train_ip_counts = df['ip'].value_counts()
train_unique_uri = df.groupby('ip')['uri'].nunique()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# Drop label column for raw X
X_raw = df.drop(columns=['label'])

# Train/test split like Colab (you can also split into train/val/test if desired)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, stratify=y, random_state=42
)

# Build fake validation split (optional). Here we keep simple: no separate val.
X_val_raw, y_val = None, None

train_df = X_train_raw.copy()
test_df = X_test_raw.copy()

# Extract features
print("Step 2: Extracting features (train/test)...")
X_train_feat = extract_features(train_df, train_ip_counts, train_unique_uri)
X_test_feat = extract_features(test_df, train_ip_counts, train_unique_uri)
print(f"  Train features: {X_train_feat.shape}")
print(f"  Test features:  {X_test_feat.shape}")

# Scale numeric features
scaler = StandardScaler()
feature_names = list(X_train_feat.columns)

X_train_scaled = scaler.fit_transform(X_train_feat[feature_names])
X_test_scaled = scaler.transform(X_test_feat[feature_names])

# Step 3: Train model using ThreatPredictionModel wrapper
print("\nStep 3: Training Gradient Boosting model...")
os.makedirs('models', exist_ok=True)

gb_model = ThreatPredictionModel('gradient_boosting')
gb_model.initialize_model(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4
)
gb_history = gb_model.train(X_train_scaled, y_train, X_val=None, y_val=None)

print("\nEvaluating on test set...")
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, gb_model.predict(X_test_scaled))
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Step 4: Save model and preprocessing artifacts
print("\nStep 4: Saving model and preprocessing artifacts...")

gb_model.save_model('models/gradient_boosting_threat_model.pkl')

preprocessor_bundle = {
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_columns': feature_names,
    'train_ip_counts': train_ip_counts,
    'train_unique_uri': train_unique_uri
}
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor_bundle, f)
print("✓ Saved preprocessor.pkl")

# Step 5: Save summary
summary = {
    'gradient_boosting': {
        'accuracy': float(test_acc),
        'training_time': gb_history['training_time']
    },
    'label_classes': list(label_encoder.classes_),
    'num_features': len(feature_names)
}
with open('models/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved models/training_summary.json")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
