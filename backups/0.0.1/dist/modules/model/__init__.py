from typing import Union
from ..model.types import IRSIConfig, IEMAConfig, IInterpreterConfig, IArimaConfig, \
    IPrediction, IPredictionMetaData, ISingleModelConfig, IModel, ITrainingDataConfig, \
        ITrainingDataActivePosition, ITrainingDataReceipt, ITrainingDataPriceActionInsight, \
            ITrainingDataPredictionInsight
from ..model.Interpreter import Interpreter
from ..model.SingleModel import SingleModel
from ..model.TrainingData import TrainingData
from ..model.MultiModel import MultiModel







# Model Factory
# Based on given configuration, it returns the appropiate Model Instance
def Model(config: IModel) -> Union[SingleModel, MultiModel]:
    """Returns the instance of a SingleModel or a MultiModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of module to return the instance of.

    Returns:
        Union[SingleModel, MultiModel]
    """
    return MultiModel(config) if len(config['single_models']) > 1 else SingleModel(config)