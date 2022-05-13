from typing import TypedDict, List, Dict
from pandas import DataFrame
from modules.keras_models import IKerasModelConfig, IKerasModelSummary, IKerasModelTrainingHistory



## Regression ##




# Regresion Configuration
# The configuration that was used to train and will predict based on.
class IRegressionConfig(TypedDict):
    # The identifier of the model
    id: str

    # Important information regarding the trained model
    description: str

    # The number of candlesticks it will lookback to make a prediction
    lookback: int

    # The number of predictions it will generate
    predictions: int

    # The summary of the KerasModel
    summary: IKerasModelSummary










## Regression Training ##





# Regression Training Configuration
# The configuration that will be used to initialize, train and save the model.
class IRegressionTrainingConfig(TypedDict):
    # The ID of the model. Must be descriptive, compatible with filesystems and preffixed with 'R_'
    id: str

    # Any relevant data that should be attached to the trained model.
    description: str

    # The number of prediction candlesticks that will look into the past in order to make a prediction.
    lookback: int

    # The number of predictions to be generated
    predictions: int

    # The learning rate to be used by the optimizer
    learning_rate: float

    # The optimizer to be used.
    optimizer: str # 'adam'|'rmsprop'

    # The loss function to be used
    loss: str # 'mse'|'mae'

    # The metric to be used for meassuring the val_loss
    metric: str # 'mse'|'mae'

    # Batch Size
    batch_size: int

    # Train Data Shuffling
    shuffle_data: bool

    # Keras Model Configuration
    keras_model: IKerasModelConfig





# Regression Training Batch
# Keras Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI
class IRegressionTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # The configurations for the models that will be trained within the batch.
    models: List[IRegressionTrainingConfig]






# Training Window Generator Configuration
# The configuration to initialize the data windowing for training.
class ITrainingWindowGeneratorConfig(TypedDict):
    # The number of consecutive inputs 
    input_width: int

    # The number of predictions to be generated (output)
    label_width: int

    # Normalized Train DF Subset
    train_df: DataFrame

    # Normalized Train DF Subset
    val_df: DataFrame

    # Normalized Test DF Subset
    test_df: DataFrame

    # List of label column names
    label_columns: List[str]

    # Batch Size
    batch_size: int

    # Train Data Shuffling
    shuffle_data: bool







# Regression Evaluation
# Evaluation performed right after the model is trained in order to get an overview of the
# potential accuracy, as well as the prediction type distribution.
# Each evaluation is performed using a random candlestick open time and is evaluated against
# the candlestick placed at the end of the window based on the model's predictions config.
class IRegressionEvaluation(TypedDict):
    # The number of evaluations performed on the Regression
    evaluations: int
    max_evaluations: int

    # The number of times the Regression predicted a price increase
    increase_num: int
    increase_successful_num: int

    # The number of times the Regression predicted a price decrease
    decrease_num: int
    decrease_successful_num: int

    # Accuracy
    increase_acc: int
    decrease_acc: int
    acc: int

    # Increase Predictions Overview
    increase_max: float
    increase_min: float
    increase_mean: float
    increase_successful_max: float
    increase_successful_min: float
    increase_successful_mean: float

    # Decrease Predictions Overview
    decrease_max: float
    decrease_min: float
    decrease_mean: float
    decrease_successful_max: float
    decrease_successful_min: float
    decrease_successful_mean: float









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
    val_size: int       # Number of rows in the val dataset
    test_size: int      # Number of rows in the test dataset

    # Data Summary - Description extracted directly from the normalized dataframe
    training_data_summary: Dict[str, Dict[str, float]]


    ## Training Configuration ##
    lookback: int
    predictions: int
    learning_rate: float
    optimizer: str
    loss: str
    metric: str
    batch_size: int    
    shuffle_data: bool
    keras_model_config: IKerasModelConfig


    ## Training ##

    # Date Range
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    training_history: IKerasModelTrainingHistory

    # Result of the evaluation of the test dataset
    test_evaluation: List[float] # [loss, metric]

    # Regression Post-Training Evaluation
    regression_evaluation: IRegressionEvaluation

    # The configuration of the Regression
    regression_config: IRegressionConfig