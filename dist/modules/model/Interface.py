from typing import Union
from pandas import DataFrame
from modules.model import IPrediction, IModel



# Model Interface
# ArimaModel, RegressionModel and ClassificationModel implement the following interface
# in order to ensure compatibility across any of the processes.
class ModelInterface:
    # Performs a prediction based on the current time
    def predict(
        self, 
        current_timestamp: int, 
        lookback_df: Union[DataFrame, None] = None, 
        enable_cache: bool = False
    ) -> IPrediction:
        raise NotImplementedError("Model.predict has not been implemented.")

    # Retrieves the lookback set on the ArimaModel
    def get_lookback(self) -> int:
        raise NotImplementedError("Model.get_lookback has not been implemented.")

    # Retrieves the configuration of the Model after being initialized
    def get_model(self) -> IModel:
        raise NotImplementedError("Model.get_model has not been implemented.")

    # Checks if a config is for the Model
    @staticmethod
    def is_config(model: IModel) -> bool:
        raise NotImplementedError("Model.is_config has not been implemented.")






