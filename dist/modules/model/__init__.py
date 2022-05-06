from typing import Union, Any
from ..model.types import IArimaModelInterpreterConfig, IArimaModelConfig, \
    IPrediction, IPredictionMetaData, IModel
from ..model.Interface import ModelInterface
from ..model.ArimaModelInterpreter import ArimaModelInterpreter
from ..model.ArimaModel import ArimaModel









# Model Factory
# Based on given configuration, it returns the appropiate Model Instance
def Model(config: IModel) -> Any:
    """Returns the instance of an ArimaModel, DecisionModel or MultiDecisionModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of module to return the instance of.

    Returns:
        Union[ArimaModel, DecisionModel, MultiDecisionModel]
    """
    # Check if it is an ArimaModel
    if ArimaModel.is_config(config):
        return ArimaModel(config)

    # Check if it is a DecisionModel
    # @TODO

    # Check if it is a MultiDecisionModel
    # @TODO

    # Otherwise, the provided configuration is invalid
    else:
        raise ValueError("Couldnt find an instance for the provided model configuration.")