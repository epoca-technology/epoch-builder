from typing import TypedDict, List, Union
from pandas import DataFrame
from modules.keras_models import IKerasModelConfig



## Regression ##




# Regresion Configuration
# ...
class IRegressionConfig(TypedDict):
    foo: str










## Regression Training ##



# Regression Training Configuration
# The configuration that will be used to initialize, train and save the model.
class IRegressionTrainingConfig(TypedDict):
    # The name of the model. 
    name: str

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




# Training History
#
class IRegressionTrainingHistory(TypedDict):
    loss: List[float]
    val_loss: List[float]
    mean_absolute_error: Union[List[float], None]
    val_mean_absolute_error: Union[List[float], None]
    mean_squared_error: Union[List[float], None]
    val_mean_squared_error: Union[List[float], None]





# Regression Training Certificate
#
class IRegressionTrainingCertificate(TypedDict):
    id: str
    name: str
    description: str