from typing import Union
from modules.types import IModel
from modules.model.KerasRegressionModel import KerasRegressionModel
from modules.model.XGBRegressionModel import XGBRegressionModel
from modules.model.KerasClassificationModel import KerasClassificationModel
from modules.model.XGBClassificationModel import XGBClassificationModel
from modules.model.ConsensusModel import ConsensusModel


# Types of Models
Model = Union[KerasRegressionModel, XGBRegressionModel, KerasClassificationModel, XGBClassificationModel, ConsensusModel]



# Model Factory
# Based on given configuration, it returns the appropiate Model Instance
def ModelFactory(config: IModel, enable_cache: bool = False) -> Model:
    """Returns the instance of a Model based on the provided configuration.

    Args:
        config: IModel
            The configuration of model to return the instance of.
        enable_cache: bool
            The state of the cache. If False, the model won't interact with the db.

    Returns:
        Model
    """
    # Check if it is a Keras Regression Model
    if KerasRegressionModel.is_config(config):
        return KerasRegressionModel(config, enable_cache=enable_cache)

    # Check if it is a XGB Regression Model
    elif XGBRegressionModel.is_config(config):
        return XGBRegressionModel(config, enable_cache=enable_cache)

    # Check if it is a Keras Classification Model
    elif KerasClassificationModel.is_config(config):
        return KerasClassificationModel(config, enable_cache=enable_cache)

    # Check if it is a XGB Classification Model
    elif XGBClassificationModel.is_config(config):
        return XGBClassificationModel(config, enable_cache=enable_cache)

    # Check if it is a ConsensusModel
    elif ConsensusModel.is_config(config):
        return ConsensusModel(config, enable_cache=enable_cache)

    # Otherwise, the provided configuration is invalid
    else:
        print(config)
        raise ValueError("Couldnt find an instance for the provided model configuration.")