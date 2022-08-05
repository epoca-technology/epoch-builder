from typing import List
from pandas import Series
from modules.types import IXGBClassificationConfig
from modules.xgb_models.Interface import XGBModelInterface



class XGBClassification(XGBModelInterface):
    """XGBClassification Class

    This class handles the initialization of a XGBoost Classification.

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        training_data_id: str
            The ID of the training data that was used to train the model.
        regressions: List[IModel]
            The list of regression models that will output the features.
        include_rsi: bool
        include_aroon: bool
            Optional Technical Analysis Features
        features_num: int
            The total number of features that will be used by the model to predict
        model: Any
            The instance of the trained model.
    """





    def __init__(self, id: str):
        """Initializes the KerasClassification Instance.

        Args:
            id: str
                The ID of the model that will be initialized.

        Raises:
            ValueError:
                If there is an issue loading the model.
                If the ID stored in the model's file is different to the one provided.
                If any of the other metadata is invalid.
        """
        raise NotImplementedError("XGBClassification.__init__ has not yet been implemented.")











    def predict(self, close_prices: Series) -> List[float]:
        """Given a list of features, it will perform a prediction.

        Args:
            features: List[float]
                The features to be used in order to generate predictions. These must match
                the classification's regressions and the technical analysis features (if any)

        Returns:
            List[float]
            [up_probability, down_probability]
        """
        raise NotImplementedError("XGBClassification.predict has not yet been implemented.")












    def get_config(self) -> IXGBClassificationConfig:
        """Returns the configuration of the XGBoost Classification Model.

        Returns:
            IXGBClassificationConfig
        """
        raise NotImplementedError("XGBClassification.get_config has not yet been implemented.")