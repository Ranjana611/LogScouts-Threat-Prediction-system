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
    Feature extraction pipeline
    DDoS, Brute Force, XSS, SQL Injection detection
    """

    def __init__(self):
        # SQL injection patterns
        self.sql_patterns = [
            r'union\s+select', r'select\s+.*\s+from', r'insert\s+into',
            r'drop\s+table', r'delete\s+from', r'update\s+.*\s+set',
            r'or\s+1\s*=\s*1', r'and\s+1\s*=\s*1', r"'\s*or\s*'",
            r'@@version', r'information_schema', r'sleep\s*\(', r'benchmark\s*\('
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
        """Parse Apache timestamp to datetime object"""
        try:
            # Remove timezone info for simplicity
            clean_ts = timestamp_str.split(' ')[0]
            return datetime.strptime(clean_ts, '%d/%b/%Y:%H:%M:%S')
        except:
            return None

    def extract_url_features(self, uri, payload):
        """Extract URL and payload based features"""
        features = {}

        # Basic URL analysis
        features['url_length'] = len(uri) if uri else 0
        features['has_parameters'] = '?' in uri if uri else False
        features['param_count'] = uri.count('&') + 1 if '?' in uri else 0

        # Path analysis
        path = uri.split('?')[0] if uri else ""
        features['path_depth'] = path.count('/') - 1
        features['is_static_resource'] = any(
            path.endswith(ext) for ext in self.static_extensions)
        features['is_auth_endpoint'] = any(
            endpoint in path.lower() for endpoint in self.auth_endpoints)

        # DVWA specific features
        features['is_dvwa_vulnerability'] = '/vulnerabilities/' in path
        #features['dvwa_module'] = self._extract_dvwa_module(path)

        # Payload analysis
        combined_payload = f"{uri} {payload}" if payload and payload != '-' else uri
        features.update(self._extract_payload_features(combined_payload))

        return features

    def _extract_dvwa_module(self, path):
        """Extract DVWA vulnerability module"""
        if '/vulnerabilities/' in path:
            parts = path.split('/vulnerabilities/')
            if len(parts) > 1:
                module = parts[1].split('/')[0]
                return module
        return 'none'

    def _extract_payload_features(self, payload_text):
        """Extract features from payload/parameters"""
        features = {}

        if not payload_text or payload_text == '-':
            return {
                'sql_injection_score': 0,
                'xss_score': 0,
                'payload_entropy': 0,
                'special_chars_count': 0,
                'encoded_payload': False
            }

        payload_lower = payload_text.lower()

        # SQL injection detection
        sql_score = sum(1 for pattern in self.sql_patterns
                        if re.search(pattern, payload_lower, re.IGNORECASE))
        features['sql_injection_score'] = sql_score

        # XSS detection
        xss_score = sum(1 for pattern in self.xss_patterns
                        if re.search(pattern, payload_lower, re.IGNORECASE))
        features['xss_score'] = xss_score

        # Payload complexity
        features['payload_entropy'] = self._calculate_entropy(payload_text)
        features['special_chars_count'] = len(
            re.findall(r'[<>"\'\(\)&=]', payload_text))
        features['encoded_payload'] = '%' in payload_text

        return features

    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0

        counter = Counter(text)
        length = len(text)
        entropy = 0

        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def extract_temporal_features(self, df, column_mapping=None, window_minutes=5):
        """Extract time-based features for DDoS and brute force detection"""

        # Handle column mapping
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

        # Check if columns exist
        if ts_col not in df.columns:
            print(
                f"⚠️  Timestamp column '{ts_col}' not found, skipping temporal features")
            return pd.DataFrame(index=df.index)

        df['parsed_timestamp'] = df[ts_col].apply(self.parse_timestamp)
        df = df.dropna(subset=['parsed_timestamp']).sort_values('parsed_timestamp')

        temporal_features = []

        for idx, row in df.iterrows():
            features = {}
            current_time = row['parsed_timestamp']
            current_ip = row['ip']

            # Define time windows
            window_start = current_time - timedelta(minutes=window_minutes)

            # Filter data in time window
            window_data = df[
                (df['parsed_timestamp'] >= window_start) &
                (df['parsed_timestamp'] <= current_time)
            ]

            # Overall request patterns
            features['requests_in_window'] = len(window_data)
            features['unique_ips_in_window'] = window_data['ip'].nunique()
            features['requests_per_ip_avg'] = len(
                window_data) / max(1, window_data['ip'].nunique())

            # Current IP behavior
            ip_data = window_data[window_data['ip'] == current_ip]
            features['requests_from_current_ip'] = len(ip_data)
            features['error_rate_current_ip'] = (
                ip_data['status'] >= 400).mean()

            # Request frequency analysis
            if len(ip_data) > 1:
                time_diffs = ip_data['parsed_timestamp'].diff(
                ).dt.total_seconds().dropna()
                features['avg_request_interval'] = time_diffs.mean()
                features['min_request_interval'] = time_diffs.min()
            else:
                features['avg_request_interval'] = float('inf')
                features['min_request_interval'] = float('inf')

            # Authentication attempts
            auth_requests = ip_data[ip_data.apply(
                lambda x: any(endpoint in str(x['uri']).lower()
                              for endpoint in self.auth_endpoints), axis=1
            )]
            features['auth_requests_in_window'] = len(auth_requests)
            features['failed_auth_rate'] = (auth_requests['status'].isin(
                [401, 403, 302])).mean() if len(auth_requests) > 0 else 0

            # Resource access patterns
            features['static_resource_ratio'] = ip_data.apply(
                lambda x: any(str(x['uri']).endswith(ext) for ext in self.static_extensions), axis=1
            ).mean()

            # DVWA vulnerability access
            vuln_requests = ip_data[ip_data['uri'].str.contains(
                '/vulnerabilities/', na=False)]
            features['vulnerability_requests'] = len(vuln_requests)

            temporal_features.append(features)

        return pd.DataFrame(temporal_features)

    def extract_user_agent_features(self, user_agent):
        """Extract features from user agent string"""
        features = {}

        if not user_agent or user_agent == '-':
            return {
                'ua_length': 0,
                'is_bot': False,
                'is_automated_tool': False,
                'browser_diversity': 0
            }

        ua_lower = user_agent.lower()

        # Basic metrics
        features['ua_length'] = len(user_agent)

        # Bot detection
        bot_indicators = ['bot', 'crawler', 'spider', 'scraper']
        features['is_bot'] = any(
            indicator in ua_lower for indicator in bot_indicators)

        # Automated tool detection
        tool_indicators = ['curl', 'wget', 'python', 'sqlmap', 'nmap', 'nikto']
        features['is_automated_tool'] = any(
            tool in ua_lower for tool in tool_indicators)

        # Browser information
        browsers = ['chrome', 'firefox', 'safari', 'edge', 'opera']
        browser_count = sum(1 for browser in browsers if browser in ua_lower)
        features['browser_diversity'] = browser_count

        return features

    def create_feature_matrix(self, df):
        
        print("Extracting URL and payload features...")

        # Handle different column name variations
        # Some CSVs might have different column names
        column_mapping = {
            'uri': ['uri', 'url', 'request', 'path'],
            'payload': ['payload', 'data', 'body'],
            'user_agent': ['user_agent', 'user-agent', 'useragent', 'ua'],
            'method': ['method', 'http_method'],
            'status': ['status', 'status_code', 'http_status'],
            'ip': ['ip', 'client_ip', 'remote_addr']
        }

        # Map columns
        mapped_cols = {}
        for standard_name, variations in column_mapping.items():
            for variation in variations:
                if variation in df.columns:
                    mapped_cols[standard_name] = variation
                    break
            if standard_name not in mapped_cols:
                print(f"⚠️  Warning: '{standard_name}' column not found")
                mapped_cols[standard_name] = None

        print(f"Column mapping: {mapped_cols}\n")

        url_features_list = []
        ua_features_list = []

        for idx, row in df.iterrows():
            # Get values with fallback
            uri = row.get(mapped_cols['uri'], '') if mapped_cols['uri'] else ''
            payload = row.get(mapped_cols['payload'],
                            '-') if mapped_cols['payload'] else '-'
            user_agent = row.get(
                mapped_cols['user_agent'], '') if mapped_cols['user_agent'] else ''

            # URL features
            url_feats = self.extract_url_features(uri, payload)
            url_features_list.append(url_feats)

            # User agent features
            ua_feats = self.extract_user_agent_features(user_agent)
            ua_features_list.append(ua_feats)

        # Convert to DataFrames
        url_features_df = pd.DataFrame(url_features_list)
        ua_features_df = pd.DataFrame(ua_features_list)

        print("Extracting temporal features...")
        temporal_features_df = self.extract_temporal_features(df, mapped_cols)

        # Combine all features
        print("Combining feature sets...")

        # Get base columns that exist
        base_cols = []
        for col in ['ip', 'timestamp', 'method', 'status', 'label']:
            if col in df.columns:
                base_cols.append(col)
            elif mapped_cols.get(col) and mapped_cols[col] in df.columns:
                base_cols.append(mapped_cols[col])

        feature_matrix = pd.concat([
            df[base_cols].reset_index(drop=True),
            url_features_df.reset_index(drop=True),
            ua_features_df.reset_index(drop=True),
            temporal_features_df.reset_index(drop=True)
        ], axis=1)

        # Add basic statistical features
        if 'status' in feature_matrix.columns:
            feature_matrix['status_class'] = feature_matrix['status'] // 100
            feature_matrix['is_error'] = feature_matrix['status'] >= 400
        elif mapped_cols['status'] in feature_matrix.columns:
            feature_matrix['status_class'] = feature_matrix[mapped_cols['status']] // 100
            feature_matrix['is_error'] = feature_matrix[mapped_cols['status']] >= 400

        if 'method' in feature_matrix.columns:
            feature_matrix['is_post_request'] = feature_matrix['method'] == 'POST'
        elif mapped_cols['method'] in feature_matrix.columns:
            feature_matrix['is_post_request'] = feature_matrix[mapped_cols['method']] == 'POST'

        return feature_matrix


# Example usage and testing
if __name__ == "__main__":
    # Sample data structure (matching your format)
    sample_data = {
        'ip': ['10.127.186.0', '10.81.43.23', '10.152.147.9'],
        'timestamp': ['01/Mar/2025:00:00:11 -0400', '01/Mar/2025:00:00:51 -0400', '01/Mar/2025:00:05:56 -0400'],
        'method': ['GET', 'GET', 'POST'],
        'uri': [
            '//dvwa/dvwa/css/main.css',
            "//dvwa/vulnerabilities/sqli/?id=1%27+UNION+SELECT+1%2C%40%40version--+-&Submit=Submit",
            '//dvwa/vulnerabilities/xss_d/?default=Spanish'
        ],
        'status': [200, 200, 302],
        'payload': ['-', "id=1%27+UNION+SELECT+1%2C%40%40version--+-&Submit=Submit", 'default=Spanish'],
        'refer': ['-', 'http://10.93.160.179//dvwa/vulnerabilities/sqli/', '-'],
        'user_agent': [
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
        ],
        'label': ['normal', 'sql_injection', 'normal']
    }

    df = pd.DataFrame(sample_data)

    # Initialize feature extractor
    extractor = ThreatFeatureExtractor()

    # Create feature matrix
    feature_matrix = extractor.create_feature_matrix(df)

    print("Feature Matrix Shape:", feature_matrix.shape)
    print("\nFeature Columns:")
    print(feature_matrix.columns.tolist())

    print("\nSample Features for SQL Injection Detection:")
    sql_row = feature_matrix[feature_matrix['label']
                             == 'sql_injection'].iloc[0]
    print(f"SQL Injection Score: {sql_row['sql_injection_score']}")
    print(f"XSS Score: {sql_row['xss_score']}")
    print(f"Payload Entropy: {sql_row['payload_entropy']:.2f}")
    print(f"Special Characters Count: {sql_row['special_chars_count']}")
    print(f"DVWA Module: {sql_row['dvwa_module']}")