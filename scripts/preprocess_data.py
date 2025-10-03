"""
Data Preprocessing Script
Prepares Apache log data for ML model training
"""
import sys
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add backend to Python path
backend_path = os.path.join(project_root, 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Change working directory to project root
os.chdir(project_root)

print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}\n")

# Import modules
try:
    from feature_extractor import ThreatFeatureExtractor
    from preprocessor import preprocessing_workflow
    import numpy as np
    import pickle
    print("✓ All modules imported successfully\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*60)
print("DATA PREPROCESSING SCRIPT")
print("="*60 + "\n")

# Initialize feature extractor
print("Step 1: Initializing feature extractor...")
feature_extractor = ThreatFeatureExtractor()
print("✓ Feature extractor initialized\n")

# Check if data file exists
data_file = 'data/raw/server_logs.csv'
if not os.path.exists(data_file):
    print(f"✗ Error: Data file not found: {data_file}")
    sys.exit(1)

# Run preprocessing
print(f"Step 2: Processing {data_file}")
print("(This may take 2-5 minutes for 24MB file)\n")

try:
    preprocessed_data = preprocessing_workflow(
        data_filepath=data_file,
        feature_extractor=feature_extractor
    )

    if preprocessed_data is None:
        print("✗ Preprocessing failed - returned None")
        sys.exit(1)

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60 + "\n")

    # Verify data exists
    print("Step 3: Verifying preprocessed data...")
    required_keys = ['X_train', 'X_val', 'X_test', 'y_train',
                     'y_val', 'y_test', 'preprocessor', 'feature_columns']

    for key in required_keys:
        if key not in preprocessed_data:
            print(f"✗ Missing key: {key}")
            sys.exit(1)

    print("✓ All required data present\n")

    # Display summary
    print("Dataset Split Summary:")
    print(f"  Training samples:   {len(preprocessed_data['X_train']):>8,}")
    print(f"  Validation samples: {len(preprocessed_data['X_val']):>8,}")
    print(f"  Testing samples:    {len(preprocessed_data['X_test']):>8,}")
    print(
        f"  Total samples:      {len(preprocessed_data['X_train']) + len(preprocessed_data['X_val']) + len(preprocessed_data['X_test']):>8,}\n")

    print(f"Feature Information:")
    print(f"  Number of features: {len(preprocessed_data['feature_columns'])}")
    print(
        f"  Feature columns: {preprocessed_data['feature_columns'][:5]}... (showing first 5)\n")

    # Display label distribution
    print("Label Distribution:")
    preprocessor = preprocessed_data['preprocessor']
    label_names = preprocessor.label_encoder.classes_
    print(f"  Classes: {list(label_names)}\n")

    # Save processed data as numpy arrays
    print("Step 4: Saving processed data to disk...")

    save_path = 'data/processed/'

    # Save training data
    np.save(os.path.join(save_path, 'X_train.npy'),
            preprocessed_data['X_train'])
    print(f"  ✓ Saved X_train.npy ({preprocessed_data['X_train'].shape})")

    np.save(os.path.join(save_path, 'y_train.npy'),
            preprocessed_data['y_train'])
    print(f"  ✓ Saved y_train.npy ({preprocessed_data['y_train'].shape})")

    # Save validation data
    np.save(os.path.join(save_path, 'X_val.npy'), preprocessed_data['X_val'])
    print(f"  ✓ Saved X_val.npy ({preprocessed_data['X_val'].shape})")

    np.save(os.path.join(save_path, 'y_val.npy'), preprocessed_data['y_val'])
    print(f"  ✓ Saved y_val.npy ({preprocessed_data['y_val'].shape})")

    # Save test data
    np.save(os.path.join(save_path, 'X_test.npy'), preprocessed_data['X_test'])
    print(f"  ✓ Saved X_test.npy ({preprocessed_data['X_test'].shape})")

    np.save(os.path.join(save_path, 'y_test.npy'), preprocessed_data['y_test'])
    print(f"  ✓ Saved y_test.npy ({preprocessed_data['y_test'].shape})")

    # Save feature column names
    with open(os.path.join(save_path, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(preprocessed_data['feature_columns'], f)
    print(f"  ✓ Saved feature_columns.pkl\n")

    # Verify files were created
    print("Step 5: Verifying saved files...")
    expected_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'feature_columns.pkl'
    ]

    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(save_path, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"  ✓ {filename} ({file_size:.2f} KB)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_exist = False

    if not all_exist:
        print("\nWarning: Some files were not created!")
    else:
        print("\n✓ All files saved successfully!")

    # Also verify preprocessor.pkl
    if os.path.exists('preprocessor.pkl'):
        print(f"  ✓ preprocessor.pkl exists in root directory\n")
    else:
        print(f"  ⚠ preprocessor.pkl not found in root directory\n")

    print("="*60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles created:")
    print("data/processed/ - Training/validation/test data")
    print("preprocessor.pkl - Preprocessor state")
    print("="*60)

except Exception as e:
    print(f"\n✗ Error during preprocessing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)