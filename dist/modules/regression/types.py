from typing import TypedDict, List, Dict
from pandas import DataFrame
from modules.keras_models import IKerasModelConfig, IKerasModelSummary, IKerasModelTrainingHistory



## Regression ##




# Regresion Configuration
# ...
class IRegressionConfig(TypedDict):
    foo: str










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

    # Keras Model Configuration
    keras_model: IKerasModelConfig




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










# Regression Training Certificate
#
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
    keras_model_config: IKerasModelConfig


    ## Training ##

    # Date Range
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    training_history: IKerasModelTrainingHistory

    # Result of the evaluation of the test dataset
    test_evaluation: List[float]

    # The summary of the KerasModel that has been trained and saved
    keras_model_summary: IKerasModelSummary