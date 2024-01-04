from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """
    Decision Tree Model.

    Inherits from BaseModel and uses the DecisionTreeClassifier algorithm
    for binary classification.

    Parameters:
    - **kwargs: Additional parameters to be passed to the
      DecisionTreeClassifier constructor.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(**kwargs)

    def get_feature_importances(self):
        """
        Get feature importances from the DecisionTreeClassifier.

        Returns:
        - np.ndarray: Feature importances based on the
          DecisionTreeClassifier's feature_importances_ attribute.
        """
        return self.model.feature_importances_
