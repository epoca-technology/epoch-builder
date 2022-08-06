from typing import List
from pandas import Series
from modules._types import IXGBRegressionConfig, XGBModelInterface



class XGBRegression(XGBModelInterface):
    """XGBRegression Class

    This class handles the initialization of a XGBoost Regression.

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        autoregressive: bool
            The type of regression.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        model: Any
            The instance of the trained model.
    """





    def __init__(self, id: str):
        """Initializes the Regression Instance.

        Args:
            id: str
                The ID of the model that will be initialized.

        Raises:
            ValueError:
                If there is an issue loading the model.
                If the ID stored in the model's file is different to the one provided.
                If any of the other metadata is invalid.
        """
        raise NotImplementedError("XGBRegression.__init__ has not yet been implemented.")











    def predict(self, close_prices: Series) -> List[float]:
        """Generates predictions based on a close price series.

        Args:
            close_prices: Series
                Lookback normalized close prices.

        Returns:
            List[float]
        """
        raise NotImplementedError("XGBRegression.predict has not yet been implemented.")












    def get_config(self) -> IXGBRegressionConfig:
        """Returns the configuration of the XGBoost Regression Model.

        Returns:
            IXGBRegressionConfig
        """
        raise NotImplementedError("XGBRegression.get_config has not yet been implemented.")