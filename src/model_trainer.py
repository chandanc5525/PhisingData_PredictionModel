import logging
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier


class ModelTrainer:

    def __init__(self):

        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1)
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):

        logging.info("Starting Model Training")

        results = []

        for name, model in self.models.items():

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, pos_label=1),
                "Recall": recall_score(y_test, y_pred, pos_label=1),
                "F1 Score": f1_score(y_test, y_pred, pos_label=1)
            })

        results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

        logging.info("Model Training Completed")
        logging.info("\n%s", results_df)

        return results_df

    def kfold_validation(self, X, y, n_splits=5):

        logging.info("Starting KFold Validation")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []

        for name, model in self.models.items():

            f1_scores = []

            for train_idx, test_idx in skf.split(X, y):

                # Clone model for each fold (VERY IMPORTANT)
                model_clone = clone(model)

                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

                model_clone.fit(X_train_fold, y_train_fold)
                y_pred = model_clone.predict(X_test_fold)

                f1_scores.append(f1_score(y_test_fold, y_pred, pos_label=1))

            results.append({
                "Model": name,
                "Average F1 Score": np.mean(f1_scores)
            })

        results_df = pd.DataFrame(results).sort_values(by="Average F1 Score", ascending=False)

        logging.info("KFold Validation Completed")
        logging.info("\n%s", results_df)

        # Retrain best model on FULL DATA before saving
        best_model_name = results_df.iloc[0]["Model"]
        best_model = clone(self.models[best_model_name])
        best_model.fit(X, y)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(best_model, "artifacts/model.pkl")

        logging.info("Best Model Saved in artifacts/model.pkl")

        return results_df