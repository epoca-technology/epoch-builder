from typing import List
from pandas import Series
from modules._types import IXGBRegressionTrainingConfig, IXGBTrainingTypeConfig
from modules.epoch.Epoch import Epoch



class XGBRegressionTraining:
    """XGBRegressionTraining Class

    This class handles the training of XGBoost Regressions.

    Class Properties:
        ...

    Instance Properties:
        ...
    """
    # Training Configuration
    TRAINING_CONFIG: IXGBTrainingTypeConfig = {
        "train_split": Epoch.TRAIN_SPLIT,
    }




    def __init__(self, config: IXGBRegressionTrainingConfig, test_mode: bool=False):
        """
        """
        raise NotImplementedError("XGBRegressionTraining.__init__ has not yet been implemented.")




