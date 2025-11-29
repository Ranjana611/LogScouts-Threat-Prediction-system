import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, unquote
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class ThreatFeatureExtractor:
    """
    Feature extraction pipeline for:
    - DDoS / high-rate traffic
    - Brute force / auth abuse
    - XSS
    - SQL Injection
    Produces a numeric-friendly feature matrix for Gradient Boosting.
    """

    def __init__(self):
        # SQL injection patterns
        self.sql_patterns = [
            r'union\s+select', r'select\s+.*\s+from', r'insert\s+into',
            r'drop\s+table', r'delete\s+from', r'update\s+.*\s+set',
            r'or\s+1\s*=\s*1', r'and\s+1\s*=\s*1', r"'\s*or\s*'",
            r'@@version', r'information_schema', r'sleep\s*\(',
            r'benchmark\s*\('
        ]

        # XSS patterns
        self.xss_patterns = [
            r'<script[^>]*>', r'javascript:', r'onclick\s*=', r'onerror\s*=',
            r'onload\s*=', r'alert\s*\(', r'document\.cookie', r'eval\s*\(',
            r'<iframe[^>]*>', r'<object[^>]*>', r'<embed[^>]*>'
        ]

        # Authentication endpoints
        self.auth_endpoints = [
            '/login', '/signin', '/auth', '/admin', '/setup', '/password'
        ]

        # Static resources
        self.static_extensions = [
            '.css', '.js', '.png', '.jpg', '.gif', '.ico', '.svg', '.woff'
        ]

    def parse_timestamp(self, timestamp_str):
        """Parse Apache timestamp to naive datetime object."""
        try:
            # Example: 01/Mar/2025:00:00:11 -0400 -> keep first part
            clean_ts = str(timestamp_str).split(' ')[0]
            return datetime.strptime(clean_ts, '%d/%b/%Y:%H:%M:%S')
        except Exception:
            return None

    def extract_url_features(self, uri, payload):
        """Extract URL and payload-based features."""
        features = {}

        uri = uri if isinstance(uri, str) else ''
        payload = payload if isinstance(payload, str) else '-'

        # Basic URL analysis
        features['url_length'] = len(uri)
        features['has_parameters'] = int('?' in uri)
        features['param_count'] = uri.count('&') + 1 if '?' in uri else 0

        # Path analysis
        path = uri.split('?', 1)[0]
        features['path_depth'] = max(path.count('/') - 1, 0)
        features['is_static_resource'] = int(
            any(path.lower().endswith(ext) for ext in self.static_extensions)
        )
        features['is_auth_endpoint'] = int(
            any(endpoint in path.lower() for endpoint in self.auth_endpoints)
        )

        # DVWA-specific
        features['is_dvwa_vulnerability'] = int('/vulnerabilities/' in path)
        features['dvwa_module'] = self._extract_dvwa_module(path)

        # Payload analysis over URI + payload
        combined_payload = f"{uri} {payload}" if payload and payload != '-' else uri
        features.update(self._extract_payload_features(combined_payload))

        return features

    def _extract_dvwa_module(self, path):
        """Extract DVWA vulnerability module name."""
        if '/vulnerabilities/' in path:
            parts = path.split('/vulnerabilities/')
            if len(parts) > 1:
                module = parts[1].split('/')[0]
                return module
        return 'none'

    def _extract_payload_features(self, payload_text):
        """Extract SQLi/XSS and complexity features from payload."""
        if not payload_text or payload_text == '-':
            return {
                'sql_injection_score': 0,
                'xss_score': 0,
                'payload_entropy': 0.0,
                'special_chars_count': 0,
                'encoded_payload': 0
            }

        features = {}
        payload_lower = str(payload_text).lower()

        # SQL injection pattern hits
        sql_score = sum(
            1 for pattern in self.sql_patterns
            if re.search(pattern, payload_lower, re.IGNORECASE)
        )
        features['sql_injection_score'] = sql_score

        # XSS pattern hits
        xss_score = sum(
            1 for pattern in self.xss_patterns
            if re.search(pattern, payload_lower, re.IGNORECASE)
        )
        features['xss_score'] = xss_score

        # Complexity and encoding
        features['payload_entropy'] = self._calculate_entropy(payload_text)
        features['special_chars_count'] = len(
            re.findall(r'[<>"\'\(\)&=]', payload_text)
        )
        features['encoded_payload'] = int('%' in payload_text)

        return features

    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0

        counter = Counter(text)
        length = len(text)
        entropy = 0.0

        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)

    def extract_temporal_features(self, df, column_mapping=None, window_minutes=5):
        """Extract time-based features for DDoS and brute-force behaviour."""
        if column_mapping is None:
            ts_col = 'timestamp'
            ip_col = 'ip'
            uri_col = 'uri'
            status_col = 'status'
        else:
            ts_col = column_mapping.get('timestamp', 'timestamp')
            ip_col = column_mapping.get('ip', 'ip')
            uri_col = column_mapping.get('uri', 'uri')
            status_col = column_mapping.get('status', 'status')

        df = df.copy()

        if ts_col not in df.columns:
            print(f"⚠️  Timestamp column '{ts_col}' not found, skipping temporal features")
            return pd.DataFrame(index=df.index)

        df['parsed_timestamp'] = df[ts_col].apply(self.parse_timestamp)
        df = df.dropna(subset=['parsed_timestamp']).sort_values('parsed_timestamp')

        temporal_features = []

        for idx, row in df.iterrows():
            features = {}
            current_time = row['parsed_timestamp']
            current_ip = row[ip_col] if ip_col in df.columns else None

            window_start = current_time - timedelta(minutes=window_minutes)

            # Window slice
            window_data = df[
                (df['parsed_timestamp'] >= window_start) &
                (df['parsed_timestamp'] <= current_time)
            ]

            # Overall request patterns
            features['requests_in_window'] = len(window_data)
            features['unique_ips_in_window'] = window_data[ip_col].nunique() if ip_col in window_data.columns else 0
            denom_ip = max(1, features['unique_ips_in_window'])
            features['requests_per_ip_avg'] = len(window_data) / denom_ip

            # Current IP behaviour
            if current_ip is not None and ip_col in window_data.columns:
                ip_data = window_data[window_data[ip_col] == current_ip]
            else:
                ip_data = window_data.iloc[0:0]  # empty

            features['requests_from_current_ip'] = len(ip_data)

            if status_col in ip_data.columns and len(ip_data) > 0:
                features['error_rate_current_ip'] = (ip_data[status_col] >= 400).mean()
            else:
                features['error_rate_current_ip'] = 0.0

            # Request frequency
            if len(ip_data) > 1:
                time_diffs = ip_data['parsed_timestamp'].diff().dt.total_seconds().dropna()
                features['avg_request_interval'] = time_diffs.mean()
                features['min_request_interval'] = time_diffs.min()
            else:
                features['avg_request_interval'] = float('inf')
                features['min_request_interval'] = float('inf')

            # Auth attempts in window
            if uri_col in ip_data.columns:
                auth_requests = ip_data[
                    ip_data[uri_col].astype(str).str.lower().apply(
                        lambda u: any(endpoint in u for endpoint in self.auth_endpoints)
                    )
                ]
                features['auth_requests_in_window'] = len(auth_requests)
                if status_col in auth_requests.columns and len(auth_requests) > 0:
                    features['failed_auth_rate'] = auth_requests[status_col].isin(
                        [401, 403, 302]
                    ).mean()
                else:
                    features['failed_auth_rate'] = 0.0
            else:
                features['auth_requests_in_window'] = 0
                features['failed_auth_rate'] = 0.0

            # Static resource ratio
            if uri_col in ip_data.columns and len(ip_data) > 0:
                static_flags = ip_data[uri_col].astype(str).apply(
                    lambda u: any(u.lower().endswith(ext) for ext in self.static_extensions)
                )
                features['static_resource_ratio'] = static_flags.mean()
            else:
                features['static_resource_ratio'] = 0.0

            # DVWA vulnerability access count
            if uri_col in ip_data.columns:
                vuln_requests = ip_data[uri_col].astype(str).str.contains(
                    '/vulnerabilities/', na=False
                )
                features['vulnerability_requests'] = vuln_requests.sum()
            else:
                features['vulnerability_requests'] = 0

            temporal_features.append(features)

        temporal_df = pd.DataFrame(temporal_features, index=df.index)
        return temporal_df

    def extract_user_agent_features(self, user_agent):
        """Extract features from the User-Agent string."""
        if not user_agent or user_agent == '-':
            return {
                'ua_length': 0,
                'is_bot': 0,
                'is_automated_tool': 0,
                'browser_diversity': 0
            }

        features = {}
        ua = str(user_agent)
        ua_lower = ua.lower()

        features['ua_length'] = len(ua)

        # Bot detection
        bot_indicators = ['bot', 'crawler', 'spider', 'scraper']
        features['is_bot'] = int(any(ind in ua_lower for ind in bot_indicators))

        # Automated tools
        tool_indicators = ['curl', 'wget', 'python', 'sqlmap', 'nmap', 'nikto']
        features['is_automated_tool'] = int(
            any(tool in ua_lower for tool in tool_indicators)
        )

        # Browser presence count
        browsers = ['chrome', 'firefox', 'safari', 'edge', 'opera']
        browser_count = sum(1 for browser in browsers if browser in ua_lower)
        features['browser_diversity'] = browser_count

        return features

    def create_feature_matrix(self, df):
        """
        Create the full feature matrix:
        - base columns: ip, timestamp, method, status, label
        - URL/payload features
        - user-agent features
        - temporal behaviour features
        """
        print("Extracting URL and payload features...")

        # Column mapping to handle different datasets
        column_mapping = {
            'uri': ['uri', 'url', 'request', 'path'],
            'payload': ['payload', 'data', 'body'],
            'user_agent': ['user_agent', 'user-agent', 'useragent', 'ua'],
            'method': ['method', 'http_method'],
            'status': ['status', 'status_code', 'http_status'],
            'ip': ['ip', 'client_ip', 'remote_addr'],
            'timestamp': ['timestamp', 'time', 'datetime']
        }

        mapped_cols = {}
        for standard_name, variations in column_mapping.items():
            mapped_cols[standard_name] = None
            for variation in variations:
                if variation in df.columns:
                    mapped_cols[standard_name] = variation
                    break
            if mapped_cols[standard_name] is None:
                print(f"⚠️  Warning: '{standard_name}' column not found")

        print(f"Column mapping: {mapped_cols}\n")

        url_features_list = []
        ua_features_list = []

        for _, row in df.iterrows():
            uri = row.get(mapped_cols['uri'], '') if mapped_cols['uri'] else ''
            payload = row.get(mapped_cols['payload'], '-') if mapped_cols['payload'] else '-'
            user_agent = row.get(mapped_cols['user_agent'], '-') if mapped_cols['user_agent'] else '-'

            url_feats = self.extract_url_features(uri, payload)
            url_features_list.append(url_feats)

            ua_feats = self.extract_user_agent_features(user_agent)
            ua_features_list.append(ua_feats)

        url_features_df = pd.DataFrame(url_features_list, index=df.index)
        ua_features_df = pd.DataFrame(ua_features_list, index=df.index)

        print("Extracting temporal features...")
        temporal_features_df = self.extract_temporal_features(df, mapped_cols)

        print("Combining feature sets...")

        # Base columns preserved for context / label
        base_cols = []
        for col in ['ip', 'timestamp', 'method', 'status', 'label']:
            if col in df.columns:
                base_cols.append(col)
            else:
                # Fallback to mapped variant
                mapped = mapped_cols.get(col)
                if mapped and mapped in df.columns:
                    base_cols.append(mapped)

        base_df = df[base_cols].copy()

        feature_matrix = pd.concat(
            [
                base_df.reset_index(drop=True),
                url_features_df.reset_index(drop=True),
                ua_features_df.reset_index(drop=True),
                temporal_features_df.reset_index(drop=True)
            ],
            axis=1
        )

        # Basic status/method derived features
        if 'status' in feature_matrix.columns:
            feature_matrix['status_class'] = feature_matrix['status'] // 100
            feature_matrix['is_error'] = feature_matrix['status'] >= 400
        elif mapped_cols['status'] and mapped_cols['status'] in feature_matrix.columns:
            col = mapped_cols['status']
            feature_matrix['status_class'] = feature_matrix[col] // 100
            feature_matrix['is_error'] = feature_matrix[col] >= 400

        if 'method' in feature_matrix.columns:
            feature_matrix['is_post_request'] = feature_matrix['method'].astype(str) == 'POST'
        elif mapped_cols['method'] and mapped_cols['method'] in feature_matrix.columns:
            col = mapped_cols['method']
            feature_matrix['is_post_request'] = feature_matrix[col].astype(str) == 'POST'

        # Ensure boolean columns are int for the model
        bool_cols = feature_matrix.select_dtypes(include=['bool']).columns
        feature_matrix[bool_cols] = feature_matrix[bool_cols].astype(int)

        return feature_matrix


if __name__ == "__main__":
    # Simple self-test with your sample
    sample_data = {
        'ip': ['10.127.186.0', '10.81.43.23', '10.152.147.9'],
        'timestamp': [
            '01/Mar/2025:00:00:11 -0400',
            '01/Mar/2025:00:00:51 -0400',
            '01/Mar/2025:00:05:56 -0400'
        ],
        'method': ['GET', 'GET', 'POST'],
        'uri': [
            '//dvwa/dvwa/css/main.css',
            "//dvwa/vulnerabilities/sqli/?id=1%27+UNION+SELECT+1%2C%40%40version--+-&Submit=Submit",
            '//dvwa/vulnerabilities/xss_d/?default=Spanish'
        ],
        'status': [200, 200, 302],
        'payload': [
            '-',
            "id=1%27+UNION+SELECT+1%2C%40%40version--+-&Submit=Submit",
            'default=Spanish'
        ],
        'refer': [
            '-',
            'http://10.93.160.179//dvwa/vulnerabilities/sqli/',
            '-'
        ],
        'user_agent': [
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
        ],
        'label': ['normal', 'sql_injection', 'normal']
    }

    df_sample = pd.DataFrame(sample_data)
    extractor = ThreatFeatureExtractor()
    fm = extractor.create_feature_matrix(df_sample)

    print("Feature Matrix Shape:", fm.shape)
    print("\nColumns:", fm.columns.tolist())
    if 'sql_injection' in fm.get('label', []):
        row = fm[fm['label'] == 'sql_injection'].iloc[0]
        print("\nSQL Injection Row Example:")
        print("  SQL Injection Score:", row['sql_injection_score'])
        print("  XSS Score:", row['xss_score'])
        print("  Payload Entropy:", row['payload_entropy'])
        print("  Special Characters Count:", row['special_chars_count'])
        print("  DVWA Module:", row['dvwa_module'])
