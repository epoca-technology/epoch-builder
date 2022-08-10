from typing import List, Dict, TypedDict
from keras import Sequential
from modules._types.model_types import IKerasRegressionConfig
from modules._types.keras_models_types import IKerasModelConfig, IKerasModelTrainingHistory, IKerasOptimizer, \
    IKerasRegressionLoss, IKerasRegressionMetric, IKerasOptimizerName
from modules._types.discovery_types import IDiscoveryPayload
from modules._types.model_evaluation_types import IModelEvaluation






## Regression Training ##





# Keras Regression Training Configuration
# The configuration that will be used to initialize, train and save the model.
class IKerasRegressionTrainingConfig(TypedDict):
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

    # The learning rate that will be used to train the model. If the value is equals to -1, the system will
    # use the InverseTimeDecay Class.
    learning_rate: float

    # The optimizer to be used.
    optimizer: IKerasOptimizer # 'adam'|'rmsprop'

    # The loss function to be used
    loss: IKerasRegressionLoss # 'mean_squared_error'|'mean_absolute_error'

    # The metric function to be used
    metric: IKerasRegressionMetric # 'mean_squared_error'|'mean_absolute_error'

    # Keras Model Configuration
    keras_model: IKerasModelConfig








# Keras Regression Training Batch
# Keras Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI.
class IKerasRegressionTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # The configurations for the models that will be trained within the batch.
    models: List[IKerasRegressionTrainingConfig]







# Keras Regression Discovery Initialization
# Once a Keras Model has been trained, it needs to be discovered prior to being evaluated.
# Therefore, the instance of the KerasRegression must be initialized prior to existing.
# When providing this configuration, the instance will use it instead of attempting to
# load the model's file
class IKerasRegressionDiscoveryInitConfig(TypedDict):
    model: Sequential
    autoregressive: bool
    lookback: int
    predictions: int







# Keras Regression Training Certificate
# Once the training, saving and evaluation completes, a certificate containing all the
# data is saved and issued for batching.
class IKerasRegressionTrainingCertificate(TypedDict):
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
    learning_rate: float
    optimizer: IKerasOptimizerName
    loss: IKerasRegressionLoss
    metric: IKerasRegressionMetric
    keras_model_config: IKerasModelConfig

    # Training
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    training_history: IKerasModelTrainingHistory

    # Result of the evaluation of the test dataset
    test_evaluation: List[float] # [loss, loss_metric]

    # Regression Discovery
    discovery: IDiscoveryPayload

    # Regression Post-Training Evaluation
    regression_evaluation: IModelEvaluation

    # The configuration of the Regression
    regression_config: IKerasRegressionConfig