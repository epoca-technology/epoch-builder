from typing import TypedDict




# Epoch Config
# The configuration to be used by the Epoch and influence the entire infrastructure.
class IEpochConfig(TypedDict):
    # Random seed to be set on all required libraries in order to guarantee reproducibility
    seed: int

    # Identifier, must be preffixed with "_". For example: "_EPOCHNAME"
    id: str

    # Simple Moving Average Window Size
    # This value will be used when initializing the candlesticks. The normalized prediction
    # dataframe is built on these values rather than normal prices.
    sma_window_size: int

    # Train Split
    # This percent value is used to separate the data that is used to train and test the 
    # models. It is important to always evaluate models on data they have not yet seen.
    # It is also important to mention that this train split does not include the validation
    # data. Moreover, This value is also used to calculate the training_evaluation range.
    train_split: float

    # Validation Split
    # This percent value is used to separate the train data from the validation data in order
    # to be able to keep track of the model's progress during training.
    validation_split: float

    # The date range of the Epoch.
    start: int
    end: int

    # The date range of the test dataset (1 - train_split)
    test_ds_start: int
    test_ds_end: int

    # Highest and lowest price within the Epoch.
    # If the price was to go above the highest or below the lowest price, trading should be
    # stopped and a new epoch should be published once the market is "stable"
    highest_price: float
    lowest_price: float

    # Regression Parameters
    # The values that represent the input and the ouput of a regression.
    # The lookback stands for the number of candlesticks from the past it needs to look at
    # in order to generate a prediction.
    # The predictions stand for the number of predictions the regressions will generate.
    regression_lookback: int
    regression_predictions: int

    # Prediction Model Evaluation
    # position_size: the total amount of USD that will be used when openning positions. The 
    #   total balance of the evaluation is equals to position_size * 2.
    # leverage: the leverage that will be used by the model evaluation to simulate real life
    #   trading.
    # idle_minutes_on_position_close: the number of minutes the prediction model will remain idle when a position is closed during the
    #   model evaluation.
    position_size: float
    leverage: int
    idle_minutes_on_position_close: int







# Default values to speed up the creation process
class IEpochDefaults(TypedDict):
    seed: int
    sma_window_size: int
    train_split: float
    validation_split: float
    epoch_width: int # Number of months that will comprise the Epoch
    regression_lookback: int
    regression_predictions: int
    position_size: float
    leverage: int
    idle_minutes_on_position_close: int
