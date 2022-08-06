from typing import Union
from modules._types import IModel
from modules.model.KerasClassificationModel import KerasClassificationModel
from modules.model.XGBClassificationModel import XGBClassificationModel


# Types of Classification Models
ClassificationModel = Union[KerasClassificationModel, XGBClassificationModel]


# Classification Model Factory
# Based on given configuration, it returns the appropiate Classification Model Instance
def ClassificationModelFactory(config: IModel, enable_cache: bool = False) -> ClassificationModel:
    """Returns the instance of an KerasRegressionModel or a XGBRegressionModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of model to return the instance of.
        enable_cache: bool
            If enabled, the model will store predictions and features in the db.

    Returns:
        ClassificationModel
    """
    # Check if it is an KerasClassificationModel
    if KerasClassificationModel.is_config(config):
        return KerasClassificationModel(config, enable_cache=enable_cache)

    # Check if it is a XGBClassificationModel
    elif XGBClassificationModel.is_config(config):
        return XGBClassificationModel(config, enable_cache=enable_cache)

    # Otherwise, the provided configuration is invalid
    else:
        print(config)
        raise ValueError("Couldnt find a classification instance for the provided model configuration.")