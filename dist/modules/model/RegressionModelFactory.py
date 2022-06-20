from typing import Union
from modules.types import IModel
from modules.model.ArimaModel import ArimaModel
from modules.model.RegressionModel import RegressionModel



# Autoregressive Regression Model Factory
# Based on given configuration, it returns the appropiate Model Instance
def RegressionModelFactory(config: IModel) -> Union[ArimaModel, RegressionModel]:
    """Returns the instance of an ArimaModel or a RegressionModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of module to return the instance of.

    Returns:
        Union[ArimaModel, RegressionModel]
    """
    # Check if it is an ArimaModel
    if ArimaModel.is_config(config):
        return ArimaModel(config)

    # Check if it is a RegressionModel
    elif RegressionModel.is_config(config):
        return RegressionModel(config)

    # Otherwise, the provided configuration is invalid
    else:
        print(config)
        raise ValueError("Couldnt find an autoregressive regression instance for the provided model configuration.")