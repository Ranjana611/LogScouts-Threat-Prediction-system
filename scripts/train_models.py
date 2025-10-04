"""
Model Training Script
Trains ML models for threat prediction
"""

import json
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
    from model_trainer import ThreatPredictionModel, train_threat_models
    from preprocessor import DataPreprocessor
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    print("✓ All modules imported successfully\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease ensure:")
    print("  1. model_trainer.py is in backend/")
    print("  2. preprocessor.py is in backend/")
    sys.exit(1)

print("="*60)
print("MODEL TRAINING SCRIPT")
print("="*60 + "\n")

# Step 1: Check if preprocessed data exists
print("Step 1: Checking for preprocessed data...")

required_files = [
    'data/processed/X_train.npy',
    'data/processed/X_val.npy',
    'data/processed/X_test.npy',
    'data/processed/y_train.npy',
    'data/processed/y_val.npy',
    'data/processed/y_test.npy',
    'preprocessor.pkl'
]

missing_files = []
for filepath in required_files:
    if not os.path.exists(filepath):
        missing_files.append(filepath)
        print(f"  ✗ Missing: {filepath}")
    else:
        print(f"  ✓ Found: {filepath}")

if missing_files:
    print(f"\n✗ Error: Missing {len(missing_files)} required files")
    print("\nPlease run preprocessing first:")
    print("  python scripts/preprocess_data.py")
    sys.exit(1)

print("\n✓ All required files found\n")

# Step 2: Load preprocessed data
print("Step 2: Loading preprocessed data...")

try:
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    y_test = np.load('data/processed/y_test.npy')

    print(f"✓ Training data loaded: {X_train.shape}")
    print(f"✓ Validation data loaded: {X_val.shape}")
    print(f"✓ Test data loaded: {X_test.shape}\n")

except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# Step 3: Load preprocessor
print("Step 3: Loading preprocessor...")

try:
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor_data = pickle.load(f)

    # Reconstruct preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.scaler = preprocessor_data['scaler']
    preprocessor.label_encoder = preprocessor_data['label_encoder']
    preprocessor.feature_columns = preprocessor_data['feature_columns']
    preprocessor.categorical_mappings = preprocessor_data['categorical_mappings']

    print(f"✓ Preprocessor loaded")
    print(f"  Label classes: {list(preprocessor.label_encoder.classes_)}")
    print(f"  Number of features: {len(preprocessor.feature_columns)}\n")

except Exception as e:
    print(f"✗ Error loading preprocessor: {e}")
    sys.exit(1)

# Step 4: Prepare data dictionary
print("Step 4: Preparing data for training...\n")

preprocessed_data = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'preprocessor': preprocessor,
    'feature_columns': preprocessor.feature_columns
}

# Step 5: Create models directory
os.makedirs('models', exist_ok=True)

# Step 6: Train Random Forest Model
print("="*60)
print("TRAINING MODEL 1: RANDOM FOREST")
print("="*60 + "\n")

rf_model = ThreatPredictionModel('random_forest')
rf_model.initialize_model(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5
)

print("Training Random Forest model...")
rf_history = rf_model.train(X_train, y_train, X_val, y_val)

print("\nEvaluating Random Forest on test set...")
rf_results = rf_model.evaluate(X_test, y_test, preprocessor.label_encoder)

# Show feature importance
print("\nTop 15 Most Important Features:")
rf_model.get_feature_importance(preprocessor.feature_columns, top_n=15)

# Save model
rf_model.save_model('models/random_forest_threat_model.pkl')
print("\n✓ Random Forest model saved\n")

# Step 7: Train Gradient Boosting Model
print("="*60)
print("TRAINING MODEL 2: GRADIENT BOOSTING")
print("="*60 + "\n")

gb_model = ThreatPredictionModel('gradient_boosting')
gb_model.initialize_model(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

print("Training Gradient Boosting model...")
gb_history = gb_model.train(X_train, y_train, X_val, y_val)

print("\nEvaluating Gradient Boosting on test set...")
gb_results = gb_model.evaluate(X_test, y_test, preprocessor.label_encoder)

# Save model
gb_model.save_model('models/gradient_boosting_threat_model.pkl')
print("\n✓ Gradient Boosting model saved\n")

# Step 8: Compare Models
print("="*60)
print("MODEL COMPARISON")
print("="*60 + "\n")

comparison_data = {
    'Model': ['Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        rf_results['accuracy'],
        gb_results['accuracy']
    ],
    'Precision (Macro)': [
        rf_results['precision_macro'],
        gb_results['precision_macro']
    ],
    'Recall (Macro)': [
        rf_results['recall_macro'],
        gb_results['recall_macro']
    ],
    'F1-Score (Macro)': [
        rf_results['f1_macro'],
        gb_results['f1_macro']
    ]
}

# Print comparison table
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("="*70)
for i in range(len(comparison_data['Model'])):
    print(f"{comparison_data['Model'][i]:<20} "
          f"{comparison_data['Accuracy'][i]*100:>10.2f}%  "
          f"{comparison_data['Precision (Macro)'][i]*100:>10.2f}%  "
          f"{comparison_data['Recall (Macro)'][i]*100:>10.2f}%  "
          f"{comparison_data['F1-Score (Macro)'][i]*100:>10.2f}%")

# Determine best model
best_idx = 0 if comparison_data['Accuracy'][0] > comparison_data['Accuracy'][1] else 1
best_model = comparison_data['Model'][best_idx]

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model}")
print(f"   Accuracy: {comparison_data['Accuracy'][best_idx]*100:.2f}%")
print(f"   F1-Score: {comparison_data['F1-Score (Macro)'][best_idx]*100:.2f}%")
print("="*60)

# Step 9: Save training summary
print("\nStep 9: Saving training summary...")

summary = {
    'random_forest': {
        'accuracy': float(rf_results['accuracy']),
        'precision': float(rf_results['precision_macro']),
        'recall': float(rf_results['recall_macro']),
        'f1_score': float(rf_results['f1_macro']),
        'training_time': rf_history['training_time']
    },
    'gradient_boosting': {
        'accuracy': float(gb_results['accuracy']),
        'precision': float(gb_results['precision_macro']),
        'recall': float(gb_results['recall_macro']),
        'f1_score': float(gb_results['f1_macro']),
        'training_time': gb_history['training_time']
    },
    'best_model': best_model,
    'label_classes': list(preprocessor.label_encoder.classes_),
    'num_features': len(preprocessor.feature_columns)
}

with open('models/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Training summary saved to models/training_summary.json\n")

# Step 10: Verify saved models
print("Step 10: Verifying saved models...")

model_files = [
    'models/random_forest_threat_model.pkl',
    'models/gradient_boosting_threat_model.pkl',
    'models/training_summary.json'
]

for filepath in model_files:
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✓ {os.path.basename(filepath)} ({file_size:.2f} KB)")
    else:
        print(f"  ✗ {os.path.basename(filepath)} NOT FOUND")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nModels saved in: models/")
print("  - random_forest_threat_model.pkl")
print("  - gradient_boosting_threat_model.pkl")
print("  - training_summary.json")
print("\nNext steps:")
print("  1. Start backend API: python backend/api.py")
print("  2. Open frontend: frontend/index.html")
print("="*60)
