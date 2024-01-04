from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseModel:
    """
    A base class for machine learning models.

    Parameters:
    - model (Union[BaseEstimator, None]):
      The underlying machine learning model.
    """

    def __init__(self, model: Union[BaseEstimator, None] = None):
        self.model = model

    def train(self, X_train, y_train):
        """
        Train the underlying machine learning model.

        Parameters:
        - X_train (pd.DataFrame): The training features.
        - y_train (pd.Series): The training labels.
        """
        self.model.fit(X_train, y_train)

    def infer(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test (pd.DataFrame): The test features.

        Returns:
        - np.ndarray: Predicted labels.
        """
        return self.model.predict(X_test)

    def get_probabilities(self, X_test):
        """
        Get class probabilities for each sample.

        Parameters:
        - X_test (pd.DataFrame): The test features.

        Returns:
        - np.ndarray: Class probabilities.
        """
        return self.model.predict_proba(X_test)

    def get_normalised_feature_importances(self):
        """
        Get normalised feature importances.

        Returns:
        - np.ndarray: Normalised feature importances.
        """
        feature_importances = self.get_feature_importances()
        return feature_importances / np.linalg.norm(feature_importances)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance using various metrics.

        Parameters:
        - X_test (pd.DataFrame): The test features.
        - y_test (pd.Series): The true labels.
        """
        y_pred = self.infer(X_test)
        y_prob = self.get_probabilities(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision: {precision_score(y_test, y_pred):.2f}")
        print(f"Recall: {recall_score(y_test, y_pred):.2f}")
        print(f"PR-AUC Score: {auc(recall, precision):.2f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")
        print()
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        feature_importances = self.get_normalised_feature_importances()
        features = X_test.columns
        # Create a DataFrame to show feature names
        # and their corresponding importances
        feature_importance_df = pd.DataFrame(
            {"Feature": features, "Importance": feature_importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )
        print(feature_importance_df)

        # Plot precision-recall curve
        plt.figure(figsize=(3, 3))
        plt.plot(recall, precision, color="darkorange", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()
