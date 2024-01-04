from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest Model.

    Inherits from BaseModel and uses the RandomForestClassifier algorithm
    for binary classification.

    Parameters:
    - **kwargs: Additional parameters to be passed to the
      RandomForestClassifier constructor.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)

    def get_feature_importances(self):
        """
        Get feature importances from the RandomForestClassifier.

        Returns:
        - np.ndarray: Feature importances based on the
          RandomForestClassifier's feature_importances_ attribute.
        """
        return self.model.feature_importances_
