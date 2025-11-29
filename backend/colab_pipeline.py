import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from datetime import datetime


# Regex patterns from your Colab notebook
sql_pattern = r"(union|select|drop|insert|--|'|or\s+1=1)"
xss_pattern = r"(<script|onerror|alert\()"


def extract_features(df, ip_counts=None, unique_uri=None):
    """
    Exact feature extraction as in the Colab notebook example you shared.
    This must be identical for both training and inference.
    """
    rows = []

    for _, r in df.iterrows():
        uri = str(r.get('uri', '') or '')
        payload = str(r.get('payload', '') or '')
        ua = str(r.get('user_agent', '') or '')
        method = str(r.get('method', '') or '')
        ip = r.get('ip', '')
        ts = r.get('timestamp')

        feature = {}

        # URL STRUCTURE
        feature['url_length'] = len(uri)
        feature['has_parameters'] = int('?' in uri)
        feature['param_count'] = uri.count('&') + 1 if '?' in uri else 0
        feature['path_depth'] = uri.count('/')
        feature['dot_count'] = uri.count('.')

        # PAYLOAD PATTERNS
        feature['payload_length'] = len(payload)
        feature['special_chars'] = sum(c in '<>"\'()=&;%' for c in payload)
        feature['encoded_chars'] = payload.count('%')
        feature['payload_sql'] = int(bool(re.search(sql_pattern, payload.lower())))
        feature['payload_xss'] = int(bool(re.search(xss_pattern, payload.lower())))

        # URI ATTACK INDICATORS
        feature['uri_sql'] = int(bool(re.search(sql_pattern, uri.lower())))
        feature['uri_xss'] = int(bool(re.search(xss_pattern, uri.lower())))

        # METHOD
        feature['is_post'] = int(method.upper() == 'POST')

        # AUTH / BRUTE SIGNAL
        feature['login_words'] = int(
            any(w in uri.lower() for w in ['login', 'signin', 'auth', 'admin', 'wp-login', 'account'])
        )
        feature['has_credentials'] = int(
            any(k in payload.lower() for k in ['username', 'user=', 'password', 'pass=', 'pwd=', 'login='])
        )

        # USER AGENT
        feature['ua_length'] = len(ua)
        feature['is_bot'] = int(
            any(tok in ua.lower() for tok in [
                'bot', 'crawler', 'spider', 'sqlmap', 'nikto', 'curl', 'python', 'wget', 'libwww'
            ])
        )

        # BEHAVIORAL STATS (based on IP)
        feature['requests_per_ip'] = int(ip_counts.get(ip, 1)) if ip_counts is not None else 1
        feature['unique_uri_per_ip'] = int(unique_uri.get(ip, 1)) if unique_uri is not None else 1

        # BRUTE-FORCE DENSITY
        feature['credential_ratio'] = feature['has_credentials'] / max(feature['requests_per_ip'], 1)

        feature['login_focus'] = int(
            feature['login_words'] == 1 and
            feature['is_post'] == 1 and
            feature['has_credentials'] == 1
        )

        # DoS PRESSURE
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

        # PAYLOAD ENTROPY
        if payload:
            probs = [payload.count(c) / len(payload) for c in set(payload)]
            feature['payload_entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            feature['payload_entropy'] = 0.0

        # NORMALITY SIGNAL
        feature['looks_human'] = int(
            feature['ua_length'] > 25 and
            feature['is_bot'] == 0 and
            feature['uri_entropy'] > 0.4 and
            feature['log_requests_per_ip'] < 3
        )

        # TIME
        if pd.notna(ts):
            feature['hour'] = ts.hour
            feature['weekday'] = ts.weekday()
        else:
            feature['hour'] = -1
            feature['weekday'] = -1

        rows.append(feature)

    return pd.DataFrame(rows)


class ColabPipeline:
    """
    Colab-like ML pipeline: feature extraction + label encoding + scaling + GB model.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.feature_names = None
        self.train_ip_counts = None
        self.train_unique_uri = None

    def fit_pipeline(self, df):
        """
        Fit the entire pipeline on the full labeled dataframe df (like in Colab).
        Saves all objects into self.
        """

        # Parse timestamp exactly as in Colab
        df = df.copy()
        df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            format='%d/%b/%Y:%H:%M:%S %z',
            errors='coerce'
        )

        # Prepare IP-level stats on full dataset (like your Colab snippet)
        self.train_ip_counts = df['ip'].value_counts()
        self.train_unique_uri = df.groupby('ip')['uri'].nunique()

        # Encode labels
        y = self.label_encoder.fit_transform(df['label'])

        # Train/test split (same parameters as in your notebook)
        X_raw = df.drop(columns=['label'])
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.25, stratify=y, random_state=42
        )

        train_df = X_train_raw.copy()
        test_df = X_test_raw.copy()

        # Extract features (train and test)
        X_train_feat = extract_features(train_df, self.train_ip_counts, self.train_unique_uri)
        X_test_feat = extract_features(test_df, self.train_ip_counts, self.train_unique_uri)

        # Scale numeric features (entire feature set is numeric already)
        self.feature_names = list(X_train_feat.columns)
        X_train_scaled = self.scaler.fit_transform(X_train_feat[self.feature_names])
        X_test_scaled = self.scaler.transform(X_test_feat[self.feature_names])

        # Fit GB model
        self.model.fit(X_train_scaled, y_train)

        # Basic training metrics on test set (optional)
        test_acc = (self.model.predict(X_test_scaled) == y_test).mean()
        print(f"Colab-style GB test accuracy inside pipeline: {test_acc*100:.2f}%")

        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }

    def transform_for_inference(self, df):
        """
        Given a new dataframe df (with ip, timestamp, method, uri, payload, user_agent),
        return X_scaled using the exact same feature extraction and scaling as during training.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            format='%d/%b/%Y:%H:%M:%S %z',
            errors='coerce'
        )

        # Use training ip stats (train_ip_counts, train_unique_uri) for these IPs;
        # unseen IPs default to 1, 1 as in training code.
        X_feat = extract_features(df, self.train_ip_counts, self.train_unique_uri)

        # Ensure all training feature columns exist
        for col in self.feature_names:
            if col not in X_feat.columns:
                X_feat[col] = 0

        X_feat = X_feat[self.feature_names]

        X_scaled = self.scaler.transform(X_feat)
        return X_scaled


def save_colab_pipeline(pipeline, model_path='models/gradient_boosting_threat_model.pkl', prep_path='preprocessor.pkl'):
    """
    Save model and preprocessing objects together.
    """
    model_data = {
        'model': pipeline.model,
        'label_encoder': pipeline.label_encoder,
        'scaler': pipeline.scaler,
        'feature_names': pipeline.feature_names,
        'train_ip_counts': pipeline.train_ip_counts,
        'train_unique_uri': pipeline.train_unique_uri
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    with open(prep_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Colab pipeline saved to {model_path} and {prep_path}")


def load_colab_pipeline(model_path='models/gradient_boosting_threat_model.pkl'):
    """
    Load model and preprocessing into a ColabPipeline instance.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    pipe = ColabPipeline()
    pipe.model = model_data['model']
    pipe.label_encoder = model_data['label_encoder']
    pipe.scaler = model_data['scaler']
    pipe.feature_names = model_data['feature_names']
    pipe.train_ip_counts = model_data['train_ip_counts']
    pipe.train_unique_uri = model_data['train_unique_uri']
    return pipe
