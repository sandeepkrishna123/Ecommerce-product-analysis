import os
import pandas as pd
import joblib
import numpy as np


class ModelLoader:
    def __init__(self):
        """Initialize model loader with paths to stored models and evaluation metrics."""
        self.model_dir = "./data/models_pkl/"
        self.results_file = "./data/dataset/model_results.csv"

        # Model filename mappings
        self.model_map = {
            "LogisticRegression": "LogisticRegression.pkl",
            "SVM": "SVM.pkl",
            "KNN": "KNN.pkl",
        }

        # Model metrics name mappings (Same as in CSV results file)
        self.metric_map = {
            "LogisticRegression": "LogisticRegression",
            "SVM": "SVM",
            "KNN": "KNN",
        }

        # Sentiment label mapping
        self.label_map = {0: "Negative", 1: "Positive"}

    def load_model(self, model_name):
        """Load a trained model from a .pkl file."""
        if model_name not in self.model_map:
            print(
                f"❌ Error: Invalid model name '{model_name}'. Available models: {list(self.model_map.keys())}"
            )
            return None

        model_path = os.path.join(self.model_dir, self.model_map[model_name])

        if os.path.exists(model_path):
            return joblib.load(model_path)  # Load model using Joblib
        else:
            print(
                f"❌ Error: Model file '{self.model_map[model_name]}' not found in {self.model_dir}!"
            )
            return None

    def predict(self, model, X_test):
        """Generate predictions using the loaded model."""
        y_test_pred = model.predict(X_test)  # Get predicted labels

        # Map labels to sentiment categories
        predictions = pd.DataFrame(
            {
                "Index": range(1, len(y_test_pred) + 1),
                "Predicted Class": y_test_pred,
                "Sentiment": [self.label_map[label] for label in y_test_pred],
            }
        )
        return predictions

    def get_model_metrics(self, model_name):
        """Retrieve stored evaluation metrics from model_results.csv."""
        if model_name not in self.metric_map:
            print(
                f"❌ Error: Invalid model name '{model_name}'. Available models: {list(self.metric_map.keys())}"
            )
            return None

        if os.path.exists(self.results_file):
            model_results = pd.read_csv(self.results_file)
            metrics = model_results[
                model_results["Model"] == self.metric_map[model_name]
            ]
            return metrics if not metrics.empty else None
        else:
            print("❌ Error: Model results file not found!")
            return None

    def predict_and_evaluate(self, model_name, X_test):
        """Load the model, make predictions, and fetch metrics."""
        model = self.load_model(model_name)
        if model:
            predictions = self.predict(model, X_test)  # Predict using the model
            metrics = self.get_model_metrics(
                model_name
            )  # Fetch model evaluation metrics
            return predictions, metrics
        else:
            return None, None
