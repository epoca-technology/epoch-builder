from typing import TypedDict, List, Union
from pandas import DataFrame




## Forecasting ##


class IForecastingConfig(TypedDict):
    foo: str







## Data Windowing ##







## Forecasting Training ##



class IForecastingTrainingConfig(TypedDict):
    # The ID of the model. 
    id: str

    # The number of prediction candlesticks that will look into the past in order to make a prediction.
    lookback: int

    # The number of predictions to be generated
    predictions: int




class ITrainingWindowGeneratorConfig(TypedDict):
    # The number of consecutive inputs 
    input_width: int

    # The number of predictions
    label_width: int

    # ...
    shift: int

    # Normalized Train DF Subset
    train_df: DataFrame

    # Normalized Train DF Subset
    val_df: DataFrame

    # Normalized Test DF Subset
    test_df: DataFrame

    # List of label column names
    label_columns: Union[List[str], None]