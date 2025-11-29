import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ThreatPredictionModel:
    """
    Machine Learning model for web threat prediction
    Uses Gradient Boosting as the only algorithm.
    """

    def __init__(self, model_type='gradient_boosting'):
        self.model_type = 'gradient_boosting'
        self.model = None
        self.training_history = {}
        self.feature_importance = None

    def initialize_model(self, **kwargs):
        """Initialize Gradient Boosting model with given hyperparameters."""
        print("Initializing gradient_boosting model...")

        self.model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 300),
            learning_rate=kwargs.get('learning_rate', 0.05),
            max_depth=kwargs.get('max_depth', 4),
            random_state=42
        )

        print("✓ gradient_boosting model initialized")
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Gradient Boosting model (keeps original signature)."""
        print("\nTraining gradient_boosting model...")
        print(f"  Training samples: {len(X_train):,}")
        if X_val is not None:
            print(f"  Validation samples: {len(X_val):,}")

        start_time = datetime.now()

        self.model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)

        print(f"\n✓ Training complete in {training_time:.2f} seconds")
        print(f"  Training Accuracy: {train_accuracy*100:.2f}%")

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

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        return self.training_history

    def evaluate(self, X_test, y_test, label_encoder):
        """Comprehensive model evaluation."""
        print("\n" + "="*60)
        print("MODEL EVALUATION (Gradient Boosting)")
        print("="*60 + "\n")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        class_names = label_encoder.classes_

        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print("--- Overall Performance ---")
        print(f"Accuracy:            {accuracy*100:.2f}%")
        print(f"Precision (Macro):   {precision_macro*100:.2f}%")
        print(f"Recall (Macro):      {recall_macro*100:.2f}%")
        print(f"F1-Score (Macro):    {f1_macro*100:.2f}%")
        print(f"\nPrecision (Weighted): {precision_weighted*100:.2f}%")
        print(f"Recall (Weighted):    {recall_weighted*100:.2f}%")
        print(f"F1-Score (Weighted):  {f1_weighted*100:.2f}%")

        print("\n--- Per-Class Performance ---")
        class_report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0,
            digits=3
        )
        print(class_report)

        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        self._print_confusion_matrix(cm, class_names)

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

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            print("\n--- ROC AUC Score ---")
            print(f"Macro Average: {roc_auc:.3f}")
        except Exception:
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
        print(f"{'Actual / Predicted':<20}", end='')
        for name in class_names:
            print(f"{name[:15]:>15}", end='')
        print()
        for i, actual in enumerate(class_names):
            print(f"{actual:<20}", end='')
            for j in range(len(class_names)):
                print(f"{cm[i][j]:>15}", end='')
            print()

    def predict(self, X, return_proba=False):
        if return_proba:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_feature_importance(self, feature_names=None, top_n=20):
        if self.feature_importance is None:
            print("Feature importance not available for this model")
            return None

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        print(f"\n--- Top {top_n} Most Important Features ---")
        for _, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<40}: {row['importance']:.4f}")

        return importance_df

    def save_model(self, filepath='models/gradient_boosting_threat_model.pkl'):
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath='models/gradient_boosting_threat_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.training_history = model_data['training_history']
        self.feature_importance = model_data.get('feature_importance')
        print(f"✓ Model loaded from {filepath}")
