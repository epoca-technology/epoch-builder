from typing import List
from pandas import Series
from modules.types import IXGBClassificationTrainingConfig, IXGBTrainingTypeConfig, IClassificationDatasets,\
    ITrainingDataFile



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
        "train_split": 0.85,
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




