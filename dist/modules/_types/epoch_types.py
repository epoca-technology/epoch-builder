from typing import TypedDict, Union




## File Paths ##




# Backtest Assets' Paths
# The paths for all the directories within the backtest_assets directory which are used
# to read and write results.
class IBacktestAssetsPath(TypedDict):
    # Backtest Assets Root Directory
    assets: str

    # Backtest Configurations
    configurations: str

    # Backtest Results
    results: str




# Model Assets' Paths
# The paths for all the directories within the model_assets directory.
class IModelAssetsPath(TypedDict):
    # Model Assets Root Directory
    assets: str

    # Batched training certificates grouped by trainable model type
    batched_training_certificates: str

    # Classification Training Data Files
    classification_training_data: str

    # Trained Models that can be used by the Software. Models that won't be exported
    # in the epoch should not be kept in this directory.
    models: str

    # All the models trained during the Epoch should be kept in this directory, grouped
    # by trainable model type.
    models_bank: str

    # Regression Selection Results
    regression_selection: str

    # Configurations generated by Hyperparams in order to train KerasClassification Models
    keras_classification_training_configs: str

    # Configurations generated by Hyperparams in order to train KerasRegression Models
    keras_regression_training_configs: str

    # Configurations generated by Hyperparams in order to train XGBClassification Models
    xgb_classification_training_configs: str

    # Configurations generated by Hyperparams in order to train XGBRegression Models
    xgb_regression_training_configs: str









## Backtest Config Factory ##











## Configuration ##



# Epoch Config
# The configuration to be used by the Epoch and influence the entire infrastructure.
class IEpochConfig(TypedDict):
    # Random seed to be set on all required libraries in order to guarantee reproducibility
    seed: int

    # Identifier, must be preffixed with "_". For example: "_EPOCHNAME"
    id: str

    # Train Split
    # This percent value is used to separate the data that is used to train and test the 
    # models. It is important to always evaluate models on data they have not yet seen.
    # This value is also used to calculate the training_evaluation range.
    train_split: float

    # The range of the Epoch. These values are used for:
    # 1) Calculate the training evaluation range (1 - train_split)
    # 2) Calculate the backtest range (epoch_width * backtest_split)
    start: int
    end: int

    # The training evaluation range is used for the following:
    # 1) Evaluate freshly trained Regression Models
    # 2) Evaluate freshly trained Classification Models
    # training_evaluation_range = 1 - train_split
    training_evaluation_start: int
    training_evaluation_end: int

    # The backtest range is used for the following:
    # 1) Discover Regressions & Classifications
    # 2) Backtest shortlisted ClassificationModels
    # 3) Backtest generated ConsensusModels
    # backtest_range = epoch_width * backtest_split
    backtest_start: int
    backtest_end: int

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

    # Model Discovery Steps
    # Similar to the Classification Training Data, when discovering Classifications or Regressions,
    # the process iterates over the prediction candlesticks based on the model_discovery_steps and
    # evaluates every prediction generated by the model, generating the discovery and the discovery
    # payload.
    model_discovery_steps: int

    # The number of minutes the model will remain idle when a position is closed during backtests
    idle_minutes_on_position_close: int

    # The identifier of the classification training data for unit tests
    classification_training_data_id_ut: Union[str, None]

    # The identifier of the selected classification training data
    classification_training_data_id: Union[str, None]





# Default values to speed up the creation process
class IEpochDefaults(TypedDict):
    seed: int
    train_split: float
    backtest_split: float
    epoch_width: int # Number of months that will comprise the Epoch
    regression_lookback: int
    regression_predictions: int
    model_discovery_steps: int
    idle_minutes_on_position_close: int
