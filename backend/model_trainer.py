import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ThreatPredictionModel:
    """
    Machine Learning model
    Supports multiple algorithms and ensemble methods
    """

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.training_history = {}
        self.feature_importance = None

    def initialize_model(self, model_type=None, **kwargs):
        """Initialize ML model based on type"""
        if model_type:
            self.model_type = model_type

        print(f"Initializing {self.model_type} model...")

        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=kwargs.get('max_depth', 15),
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                multi_class='multinomial'
            ),
            'svm': SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights='distance'
            )
        }

        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = models[self.model_type]
        print(f"✓ {self.model_type} model initialized")

        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print(f"\nTraining {self.model_type} model...")
        print(f"  Training samples: {len(X_train):,}")

        if X_val is not None:
            print(f"  Validation samples: {len(X_val):,}")

        start_time = datetime.now()

        # Train model
        self.model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        # Training metrics
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)

        print(f"\n✓ Training complete in {training_time:.2f} seconds")
        print(f"  Training Accuracy: {train_accuracy*100:.2f}%")

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"  Validation Accuracy: {val_accuracy*100:.2f}%")

            self.training_history = {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'training_time': training_time,
                'model_type': self.model_type
            }
        else:
            self.training_history = {
                'train_accuracy': train_accuracy,
                'training_time': training_time,
                'model_type': self.model_type
            }

        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        return self.training_history

    def evaluate(self, X_test, y_test, label_encoder):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60 + "\n")

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Get class names
        class_names = label_encoder.classes_

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Multi-class metrics
        precision_macro = precision_score(
            y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(
            y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(
            y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(
            y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(
            y_test, y_pred, average='weighted', zero_division=0)

        print("--- Overall Performance ---")
        print(f"Accuracy:           {accuracy*100:.2f}%")
        print(f"Precision (Macro):  {precision_macro*100:.2f}%")
        print(f"Recall (Macro):     {recall_macro*100:.2f}%")
        print(f"F1-Score (Macro):   {f1_macro*100:.2f}%")
        print(f"\nPrecision (Weighted): {precision_weighted*100:.2f}%")
        print(f"Recall (Weighted):    {recall_weighted*100:.2f}%")
        print(f"F1-Score (Weighted):  {f1_weighted*100:.2f}%")

        # Per-class metrics
        print("\n--- Per-Class Performance ---")
        class_report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0,
            digits=3
        )
        print(class_report)

        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        self._print_confusion_matrix(cm, class_names)

        # Per-class detailed analysis
        print("\n--- Detailed Threat Detection Analysis ---")
        for i, threat_type in enumerate(class_names):
            threat_mask = (y_test == i)
            if threat_mask.sum() > 0:
                threat_accuracy = (y_pred[threat_mask] == i).mean()
                false_negatives = ((y_test == i) & (y_pred != i)).sum()
                false_positives = ((y_test != i) & (y_pred == i)).sum()

                print(f"\n{threat_type.upper()}:")
                print(f"  Detection Rate:    {threat_accuracy*100:.2f}%")
                print(f"  False Negatives:   {false_negatives}")
                print(f"  False Positives:   {false_positives}")

        # ROC AUC for multi-class
        try:
            roc_auc = roc_auc_score(
                y_test, y_pred_proba, multi_class='ovr', average='macro')
            print(f"\n--- ROC AUC Score ---")
            print(f"Macro Average: {roc_auc:.3f}")
        except:
            print("\n--- ROC AUC Score ---")
            print("Not available for this configuration")

        print("\n" + "="*60 + "\n")

        evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }

        return evaluation_results

    def _print_confusion_matrix(self, cm, class_names):
        """Print formatted confusion matrix"""
        # Header
        print(f"{'Actual / Predicted':<20}", end='')
        for name in class_names:
            print(f"{name[:15]:>15}", end='')
        print()

        # Rows
        for i, actual in enumerate(class_names):
            print(f"{actual:<20}", end='')
            for j in range(len(class_names)):
                print(f"{cm[i][j]:>15}", end='')
            print()

    def predict(self, X, return_proba=False):
        """Make predictions on new data"""
        if return_proba:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get and display feature importance"""
        if self.feature_importance is None:
            print("Feature importance not available for this model")
            return None

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(
                len(self.feature_importance))]

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        print(f"\n--- Top {top_n} Most Important Features ---")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<40}: {row['importance']:.4f}")

        return importance_df

    def save_model(self, filepath='threat_model.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath='threat_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.training_history = model_data['training_history']
        self.feature_importance = model_data.get('feature_importance')

        print(f"✓ Model loaded from {filepath}")


class EnsemblePredictor:
    """
    Ensemble model combining multiple classifiers
    Uses voting mechanism for robust predictions
    """

    def __init__(self):
        self.models = {}
        self.weights = {}

    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        print(f"✓ Added {name} to ensemble (weight: {weight})")

    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in the ensemble"""
        print("\n" + "="*60)
        print("ENSEMBLE TRAINING")
        print("="*60 + "\n")

        results = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            history = model.train(X_train, y_train, X_val, y_val)
            results[name] = history

        print("\n" + "="*60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*60 + "\n")

        return results

    def predict(self, X, method='weighted_voting'):
        """Ensemble prediction using voting"""
        predictions = {}
        probabilities = {}

        # Get predictions from all models
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
            probabilities[name] = model.predict(X, return_proba=True)

        if method == 'weighted_voting':
            # Weighted probability averaging
            weighted_proba = np.zeros_like(
                probabilities[list(self.models.keys())[0]])
            total_weight = sum(self.weights.values())

            for name, proba in probabilities.items():
                weighted_proba += proba * (self.weights[name] / total_weight)

            final_predictions = np.argmax(weighted_proba, axis=1)

        elif method == 'majority_voting':
            # Simple majority voting
            pred_array = np.array([predictions[name]
                                  for name in self.models.keys()])
            final_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=pred_array
            )

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return final_predictions


# Training workflow
def train_threat_models(preprocessed_data):
    """
    Complete training workflow for threat prediction models
    """
    print("\n" + "="*60)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*60 + "\n")

    # Unpack data
    X_train = preprocessed_data['X_train']
    X_val = preprocessed_data['X_val']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_val = preprocessed_data['y_val']
    y_test = preprocessed_data['y_test']
    preprocessor = preprocessed_data['preprocessor']
    feature_columns = preprocessed_data['feature_columns']

    # Train Random Forest (Primary Model)
    print("="*60)
    print("Training Primary Model: Random Forest")
    print("="*60)

    rf_model = ThreatPredictionModel('random_forest')
    rf_model.initialize_model(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5
    )
    rf_model.train(X_train, y_train, X_val, y_val)

    # Evaluate Random Forest
    print("\nEvaluating Random Forest on Test Set...")
    rf_results = rf_model.evaluate(X_test, y_test, preprocessor.label_encoder)

    # Feature importance
    rf_model.get_feature_importance(feature_columns, top_n=15)

    # Save primary model
    rf_model.save_model('models/random_forest_threat_model.pkl')

    # Train Gradient Boosting (Secondary Model)
    print("\n" + "="*60)
    print("Training Secondary Model: Gradient Boosting")
    print("="*60)

    gb_model = ThreatPredictionModel('gradient_boosting')
    gb_model.initialize_model(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    gb_model.train(X_train, y_train, X_val, y_val)

    # Evaluate Gradient Boosting
    print("\nEvaluating Gradient Boosting on Test Set...")
    gb_results = gb_model.evaluate(X_test, y_test, preprocessor.label_encoder)

    # Save secondary model
    gb_model.save_model('models/gradient_boosting_threat_model.pkl')

    # Create Ensemble Model
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60 + "\n")

    ensemble = EnsemblePredictor()
    ensemble.add_model('random_forest', rf_model, weight=0.6)
    ensemble.add_model('gradient_boosting', gb_model, weight=0.4)

    # Evaluate Ensemble
    print("\nEvaluating Ensemble Model...")
    ensemble_predictions = ensemble.predict(X_test, method='weighted_voting')
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

    print(f"\n✓ Ensemble Model Accuracy: {ensemble_accuracy*100:.2f}%")

    # Compare all models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")

    comparison = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Ensemble'],
        'Accuracy': [
            rf_results['accuracy'],
            gb_results['accuracy'],
            ensemble_accuracy
        ],
        'F1-Score (Macro)': [
            rf_results['f1_macro'],
            gb_results['f1_macro'],
            f1_score(y_test, ensemble_predictions, average='macro')
        ],
        'Precision (Macro)': [
            rf_results['precision_macro'],
            gb_results['precision_macro'],
            precision_score(y_test, ensemble_predictions, average='macro')
        ],
        'Recall (Macro)': [
            rf_results['recall_macro'],
            gb_results['recall_macro'],
            recall_score(y_test, ensemble_predictions, average='macro')
        ]
    })

    print(comparison.to_string(index=False))

    # Select best model
    best_model_idx = comparison['Accuracy'].idxmax()
    best_model_name = comparison.loc[best_model_idx, 'Model']

    print(f"\n🏆 Best Model: {best_model_name}")
    print(
        f"   Accuracy: {comparison.loc[best_model_idx, 'Accuracy']*100:.2f}%")

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60 + "\n")

    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'ensemble': ensemble,
        'best_model': best_model_name,
        'comparison': comparison,
        'rf_results': rf_results,
        'gb_results': gb_results
    }


# Example usage
if __name__ == "__main__":
    print("ML Model Training Pipeline Ready!")
    print("\nSupported Models:")
    print("  - Random Forest (Primary)")
    print("  - Gradient Boosting (Secondary)")
    print("  - Ensemble (Combined)")
    print("\nThreat Types Detected:")
    print("  - Normal Traffic")
    print("  - DDoS Attacks")
    print("  - Brute Force Attacks")
    print("  - Cross-Site Scripting (XSS)")
    print("  - SQL Injection")
