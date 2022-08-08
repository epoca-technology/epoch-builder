from typing import List
from pandas import Series
from modules._types import IXGBClassificationTrainingConfig, IXGBTrainingTypeConfig, IClassificationDatasets,\
    ITrainingDataFile
from modules.epoch.Epoch import Epoch



class XGBClassificationTraining:
    """XGBClassificationTraining Class

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




    def __init__(
        self, 
        training_data_file: ITrainingDataFile, 
        config: IXGBClassificationTrainingConfig, 
        datasets: IClassificationDatasets,
        test_mode: bool=False
    ):
        """
        """
        raise NotImplementedError("XGBClassificationTraining.__init__ has not yet been implemented.")




