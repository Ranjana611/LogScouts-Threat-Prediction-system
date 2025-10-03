import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import pickle
import json
from datetime import datetime


class DataPreprocessor:
    """
    Data preprocessing pipeline.
    Handles data loading, cleaning, feature scaling, and train/test splitting
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_mappings = {}

    def load_dataset(self, filepath, delimiter=None):
        """
        Load dataset from CSV file
        Auto-detects delimiter if not specified
        """
        try:
            # Try to detect delimiter automatically
            if delimiter is None:
                # Read first line to detect delimiter
                with open(filepath, 'r') as f:
                    first_line = f.readline()

                # Check which delimiter is present
                if '\t' in first_line:
                    delimiter = '\t'
                    print("  Detected delimiter: TAB")
                elif ',' in first_line:
                    delimiter = ','
                    print("  Detected delimiter: COMMA")
                else:
                    delimiter = ','
                    print("  Using default delimiter: COMMA")

            # Load with detected delimiter
            df = pd.read_csv(filepath, delimiter=delimiter, low_memory=False)

            # Check if columns were parsed correctly
            if len(df.columns) == 1:
                # Single column means wrong delimiter, try other options
                print("  ⚠ Single column detected, trying alternative delimiters...")

                for delim in ['\t', ',', ';', '|', ' ']:
                    try:
                        df = pd.read_csv(
                            filepath, delimiter=delim, low_memory=False)
                        if len(df.columns) > 1:
                            print(
                                f"  ✓ Successfully parsed with delimiter: {repr(delim)}")
                            break
                    except:
                        continue

            print(f"✓ Dataset loaded successfully")
            print(f"  Total records: {len(df):,}")
            print(f"  Columns found: {len(df.columns)}")
            print(f"  Column names: {df.columns.tolist()}\n")

            # Basic validation
            required_columns = ['ip', 'timestamp',
                                'method', 'uri', 'status', 'label']
            missing_columns = [
                col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"⚠ Warning: Missing columns: {missing_columns}")
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None

    def analyze_dataset(self, df):
        """Perform exploratory analysis on the dataset"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)

        # Basic statistics
        print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(
            f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Label distribution
        print("\n--- Label Distribution ---")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{label:20s}: {count:6d} ({percentage:5.2f}%)")

        # Check for class imbalance
        max_class = label_counts.max()
        min_class = label_counts.min()
        imbalance_ratio = max_class / \
            min_class if min_class > 0 else float('inf')
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 3:
            print("Warning: Significant class imbalance detected!")
            print("  Recommendation: Consider using class weights or SMOTE")

        # Time span analysis
        print("\n--- Temporal Coverage ---")
        try:
            df['temp_timestamp'] = pd.to_datetime(
                df['timestamp'].str.split(' ').str[0],
                format='%d/%b/%Y:%H:%M:%S'
            )
            print(f"Start Date: {df['temp_timestamp'].min()}")
            print(f"End Date:   {df['temp_timestamp'].max()}")
            print(
                f"Time Span:  {(df['temp_timestamp'].max() - df['temp_timestamp'].min()).days} days")
            df.drop('temp_timestamp', axis=1, inplace=True)
        except:
            print("Unable to parse timestamps")

        # IP statistics
        print("\n--- IP Address Statistics ---")
        print(f"Unique IPs: {df['ip'].nunique():,}")
        print(f"Top 5 Most Active IPs:")
        top_ips = df['ip'].value_counts().head()
        for ip, count in top_ips.items():
            print(f"  {ip}: {count:,} requests")

        # HTTP Method distribution
        print("\n--- HTTP Methods ---")
        method_counts = df['method'].value_counts()
        for method, count in method_counts.items():
            print(f"{method}: {count:,}")

        # Status code distribution
        print("\n--- Status Codes ---")
        status_counts = df['status'].value_counts().sort_index()
        for status, count in status_counts.head(10).items():
            print(f"{status}: {count:,}")

        # Missing values
        print("\n--- Missing Values ---")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            for col, count in missing[missing > 0].items():
                print(f"{col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print("No missing values detected ✓")

        print("\n" + "="*60 + "\n")

        return {
            'total_records': len(df),
            'label_distribution': label_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'unique_ips': df['ip'].nunique()
        }

    def clean_data(self, df):
        """Clean and prepare data for feature extraction"""
        print("Cleaning data...")

        df_clean = df.copy()

        # Handle missing values
        df_clean['payload'] = df_clean['payload'].fillna('-')
        df_clean['refer'] = df_clean['refer'].fillna('-')
        df_clean['user_agent'] = df_clean['user_agent'].fillna('-')

        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_count - len(df_clean)
        if removed > 0:
            print(f"  Removed {removed} duplicate records")

        # Validate status codes
        df_clean = df_clean[df_clean['status'].between(100, 599)]

        # Remove records with invalid timestamps
        df_clean = df_clean[df_clean['timestamp'].notna()]

        print(f"✓ Data cleaning complete: {len(df_clean):,} records remaining")

        return df_clean

    def prepare_features(self, feature_matrix, is_training=True):
        """
        Prepare features for machine learning
        - Encode categorical variables
        - Scale numerical features
        - Handle infinite values
        """
        print("Preparing features for ML...")

        df = feature_matrix.copy()

        # Identify feature columns (exclude metadata and label)
        metadata_cols = ['ip', 'timestamp', 'label', 'dvwa_module']
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        if is_training:
            self.feature_columns = feature_cols

        # Encode categorical features
        categorical_features = df[feature_cols].select_dtypes(
            include=['object', 'bool']).columns

        for col in categorical_features:
            if is_training:
                # Create mapping for training
                unique_values = df[col].unique()
                self.categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(unique_values)}

            # Apply mapping
            df[col] = df[col].map(
                self.categorical_mappings[col]).fillna(-1).astype(int)

        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with median (for numerical features)
        numerical_features = df[feature_cols].select_dtypes(
            include=[np.number]).columns
        df[numerical_features] = df[numerical_features].fillna(
            df[numerical_features].median())

        # Scale features
        if is_training:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        print(f"✓ Features prepared: {len(feature_cols)} features")

        return df, feature_cols

    def encode_labels(self, labels, is_training=True):
        """Encode threat labels to numerical format"""
        if is_training:
            encoded_labels = self.label_encoder.fit_transform(labels)
            print(
                f"✓ Label encoding: {len(self.label_encoder.classes_)} classes")
            print(f"  Classes: {list(self.label_encoder.classes_)}")
        else:
            encoded_labels = self.label_encoder.transform(labels)

        return encoded_labels

    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into training, validation, and test sets
        Maintains temporal order for time-series aware splitting
        """
        print(f"\nSplitting dataset...")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )

        print(f"✓ Dataset split complete:")
        print(
            f"  Training:   {len(X_train):6,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(
            f"  Validation: {len(X_val):6,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(
            f"  Testing:    {len(X_test):6,} samples ({len(X_test)/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def calculate_class_weights(self, y):
        """Calculate class weights to handle imbalanced datasets"""
        classes = np.unique(y)
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )

        class_weight_dict = dict(zip(classes, weights))

        print("\n--- Class Weights ---")
        for class_idx, weight in class_weight_dict.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"{class_name:20s}: {weight:.3f}")

        return class_weight_dict

    def save_preprocessor(self, filepath='preprocessor.pkl'):
        """Save preprocessor state for later use"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'categorical_mappings': self.categorical_mappings
        }

        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        print(f"✓ Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath='preprocessor.pkl'):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.scaler = preprocessor_data['scaler']
        self.label_encoder = preprocessor_data['label_encoder']
        self.feature_columns = preprocessor_data['feature_columns']
        self.categorical_mappings = preprocessor_data['categorical_mappings']

        print(f"✓ Preprocessor loaded from {filepath}")

# Complete preprocessing workflow


def preprocessing_workflow(data_filepath, feature_extractor):
    """
    Complete data preprocessing workflow
    Returns prepared data ready for ML training
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING WORKFLOW")
    print("="*60 + "\n")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    df = preprocessor.load_dataset(data_filepath)
    if df is None:
        return None

    # Step 2: Analyze dataset
    print("\nStep 2: Analyzing dataset...")
    stats = preprocessor.analyze_dataset(df)

    # Step 3: Clean data
    print("\nStep 3: Cleaning data...")
    df_clean = preprocessor.clean_data(df)

    # Step 4: Extract features
    print("\nStep 4: Extracting features...")
    feature_matrix = feature_extractor.create_feature_matrix(df_clean)

    # Step 5: Prepare features for ML
    print("\nStep 5: Preparing features for ML...")
    feature_matrix_prepared, feature_cols = preprocessor.prepare_features(
        feature_matrix, is_training=True
    )

    # Step 6: Encode labels
    print("\nStep 6: Encoding labels...")
    y = preprocessor.encode_labels(
        feature_matrix_prepared['label'], is_training=True)
    X = feature_matrix_prepared[feature_cols].values

    # Step 7: Split dataset
    print("\nStep 7: Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(
        X, y)

    # Step 8: Calculate class weights
    print("\nStep 8: Calculating class weights...")
    class_weights = preprocessor.calculate_class_weights(y_train)

    # Step 9: Save preprocessor
    print("\nStep 9: Saving preprocessor...")
    preprocessor.save_preprocessor()

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60 + "\n")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'class_weights': class_weights,
        'preprocessor': preprocessor,
        'feature_columns': feature_cols,
        'dataset_stats': stats
    }


# Example usage
if __name__ == "__main__":
    print("Data Preprocessing Pipeline Ready!")
    print("\nUsage:")
    print("1. Load your dataset")
    print("2. Run feature extraction")
    print("3. Execute preprocessing workflow")
    print("4. Data will be ready for ML model training")
