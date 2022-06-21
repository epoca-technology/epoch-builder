from typing import TypedDict, List, Dict, Union
from modules.types import IRegressionConfig, IKerasModelConfig, IKerasModelTrainingHistory, IRegressionConfig, IModelEvaluation







## Regression Training ##





# Regression Training Configuration
# The configuration that will be used to initialize, train and save the model.
class IRegressionTrainingConfig(TypedDict):
    # The ID of the model. Must be descriptive, compatible with filesystems and preffixed with 'R_'
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

    # The optimizer to be used.
    optimizer: str # 'adam'|'rmsprop'

    # The loss function to be used
    loss: str # 'mean_squared_error'|'mean_absolute_error'

    # Keras Model Configuration
    keras_model: IKerasModelConfig








# Regression Training Batch
# Keras Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI
class IRegressionTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # Start and end time - If none provided, will use all the available data
    start: Union[str, int, None]
    end: Union[str, int, None]

    # If enabled, it will delete the model files once the evaluation is complete
    hyperparams_mode: bool

    # The configurations for the models that will be trained within the batch.
    models: List[IRegressionTrainingConfig]








# Regression Training Certificate
# Once the training, saving and evaluation completes, a certificate containing all the
# data is saved and issued for batching.
class IRegressionTrainingCertificate(TypedDict):
    ## Identification ##
    id: str
    description: str


    ## Training Data ##

    # Date Range
    training_data_start: int    # Open Time of the first prediction candlestick
    training_data_end: int      # Close Time of the last prediction candlestick

    # Dataset Sizes
    train_size: int     # Number of rows in the train dataset
    test_size: int      # Number of rows in the test dataset

    # Data Summary - Description extracted directly from the normalized dataframe
    training_data_summary: Dict[str, float]


    ## Training Configuration ##
    autoregressive: bool
    lookback: int
    predictions: int
    optimizer: str
    loss: str
    keras_model_config: IKerasModelConfig


    ## Training ##

    # Date Range
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    training_history: IKerasModelTrainingHistory

    # Result of the evaluation of the test dataset
    test_evaluation: float # loss

    # Regression Post-Training Evaluation
    regression_evaluation: IModelEvaluation

    # The configuration of the Regression
    regression_config: IRegressionConfig