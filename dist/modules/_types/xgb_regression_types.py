from typing import List, TypedDict, Dict, Any
from modules._types.model_types import IXGBRegressionConfig
from modules._types.discovery_types import IDiscoveryPayload
from modules._types.model_evaluation_types import IModelEvaluation






## Regression Training ##





# XGBoost Regression Training Configuration
# The configuration that will be used to initialize, train and save the model.
class IXGBRegressionTrainingConfig(TypedDict):
    # The ID of the model.
    id: str

    # Any relevant data that should be attached to the trained model.
    description: str

    # Regression Model Type
    # Default: will generate all predictions in one go.
    # Autoregressive: will generate 1 prediction at a time and feed it to itself as an input 
    autoregressive: bool

    # The number of prediction candlesticks that will look into the past in order to make a prediction.
    lookback: int

    # The number of predictions to be generated
    predictions: int

    # @TODO








# XGBoost Regression Training Batch
# XGBoost Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI.
class IXGBRegressionTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # The configurations for the models that will be trained within the batch.
    models: List[IXGBRegressionTrainingConfig]






# XGBoost Regression Discovery Initialization
# Once a XGB Model has been trained, it needs to be discovered prior to being evaluated.
# Therefore, the instance of the XGBRegression must be initialized prior to existing.
# When providing this configuration, the instance will use it instead of attempting to
# load the model's file
class IXGBRegressionDiscoveryInitConfig(TypedDict):
    model: Any
    autoregressive: bool
    lookback: int
    predictions: int






# XGBoost Regression Training Certificate
# Once the training, saving and evaluation completes, a certificate containing all the
# data is saved and issued for batching.
class IXGBRegressionTrainingCertificate(TypedDict):
    # Identification
    id: str
    description: str

    # Training Data Date Range
    training_data_start: int    # Open Time of the first prediction candlestick
    training_data_end: int      # Close Time of the last prediction candlestick

    # Training Data Dataset Sizes
    train_size: int     # Number of rows in the train dataset
    test_size: int      # Number of rows in the test dataset

    # Training Data Summary - Description extracted directly from the normalized dataframe
    training_data_summary: Dict[str, float]

    # Training Configuration
    autoregressive: bool
    lookback: int
    predictions: int
    # @TODO

    # Training performance by epoch
    # @TODO

    # Result of the evaluation of the test dataset
    # @TODO

    # Regression Discovery
    discovery: IDiscoveryPayload

    # Regression Post-Training Evaluation
    regression_evaluation: IModelEvaluation

    # The configuration of the Regression
    regression_config: IXGBRegressionConfig