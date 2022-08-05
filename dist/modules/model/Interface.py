from typing import Union
from pandas import DataFrame
from modules.types import IPrediction, IModel



# Model Interface
# KerasClassificationModel, XGBClassificationModel and ConsensusModel implement the following interface
# in order to ensure compatibility across any of the processes.
class ModelInterface:
    # Init
    def __init__(self, config: IModel, enable_cache: bool = False):
        raise NotImplementedError("Model.__init__ has not been implemented.")

    # Performs a prediction based on the current time
    def predict(self, current_timestamp: int, lookback_df: Union[DataFrame, None] = None) -> IPrediction:
        raise NotImplementedError("Model.predict has not been implemented.")

    # Retrieves the lookback set on the Model
    def get_lookback(self) -> int:
        raise NotImplementedError("Model.get_lookback has not been implemented.")

    # Retrieves the configuration of the Model after being initialized
    def get_model(self) -> IModel:
        raise NotImplementedError("Model.get_model has not been implemented.")

    # Checks if a config is for the Model
    @staticmethod
    def is_config(model: IModel) -> bool:
        raise NotImplementedError("Model.is_config has not been implemented.")





# Regression Model Interface
# KerasRegressionModel and XGBRegressionModel implement the following interface
# in order to ensure compatibility across any of the processes.
class RegressionModelInterface(ModelInterface):
    # Generates a feature based on the current time
    def feature(self, current_timestamp: int, lookback_df: Union[DataFrame, None] = None) -> float:
        raise NotImplementedError("RegressionModel.feature has not been implemented.")