import numpy as np
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression Model.

    Inherits from BaseModel and uses the Logistic Regression algorithm
    for binary classification.

    Parameters:
    - **kwargs: Additional parameters to be passed to the
      LogisticRegression constructor.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(**kwargs)

    def get_feature_importances(self):
        """
        Get the absolute values of coefficients as feature importances.

        Returns:
        - np.ndarray: Feature importances based on
          absolute values of coefficients.
        """
        return np.abs(self.model.coef_[0])
