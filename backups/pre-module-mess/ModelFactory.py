from typing import Union
from modules.types import IModel
from modules.model.ArimaModel import ArimaModel
from modules.model.RegressionModel import RegressionModel
from modules.model.ClassificationModel import ClassificationModel



# Model Factory
# Based on given configuration, it returns the appropiate Model Instance
def ModelFactory(config: IModel) -> Union[ArimaModel, RegressionModel, ClassificationModel]:
    """Returns the instance of an ArimaModel, RegressionModel or ClassificationModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of module to return the instance of.

    Returns:
        Union[ArimaModel, RegressionModel, ClassificationModel]
    """
    # Check if it is an ArimaModel
    if ArimaModel.is_config(config):
        return ArimaModel(config)

    # Check if it is a RegressionModel
    elif RegressionModel.is_config(config):
        return RegressionModel(config)

    # Check if it is a ClassificationModel
    elif ClassificationModel.is_config(config):
        return ClassificationModel(config)

    # Otherwise, the provided configuration is invalid
    else:
        print(config)
        raise ValueError("Couldnt find an instance for the provided model configuration.")