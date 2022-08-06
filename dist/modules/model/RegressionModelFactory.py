from typing import Union
from modules._types import IModel
from modules.model.KerasRegressionModel import KerasRegressionModel
from modules.model.XGBRegressionModel import XGBRegressionModel


# Types of Regression Models
RegressionModel = Union[KerasRegressionModel, XGBRegressionModel]


# Regression Model Factory
# Based on given configuration, it returns the appropiate Regression Model Instance
def RegressionModelFactory(config: IModel, enable_cache: bool = False) -> RegressionModel:
    """Returns the instance of an KerasRegressionModel or a XGBRegressionModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of model to return the instance of.
        enable_cache: bool
            If enabled, the model will store predictions and features in the db.

    Returns:
        RegressionModel
    """
    # Check if it is an KerasRegressionModel
    if KerasRegressionModel.is_config(config):
        return KerasRegressionModel(config, enable_cache=enable_cache)

    # Check if it is a XGBRegressionModel
    elif XGBRegressionModel.is_config(config):
        return XGBRegressionModel(config, enable_cache=enable_cache)

    # Otherwise, the provided configuration is invalid
    else:
        print(config)
        raise ValueError("Couldnt find a regression instance for the provided model configuration.")