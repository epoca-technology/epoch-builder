from typing import TypedDict, Union, Literal, Dict




## File Paths ##


# Configuration Files' Paths
# The Paths for all the configuration files that are used as input for many of the modules.
class IConfigPath(TypedDict):
    epoch: str
    backtest: str
    classification_training: str
    classification_training_data: str
    regression_training: str




# Backtest Assets' Paths
# The paths for all the directories within the backtest_assets directory.
class IBacktestAssetsPath(TypedDict):
    assets: str
    configurations: str
    results: str
    regression_selection: str




# Model Assets' Paths
# The paths for all the directories within the model_assets directory.
class IModelAssetsPath(TypedDict):
    assets: str
    batched_training_certificates: str
    classification_training_configs: str
    classification_training_data: str
    classification_training_data_configs: str
    models: str
    models_bank: str
    regression_training_configs: str









## Position Exit Combinations ##




# Identifiers
IPositionExitCombinationID = Literal[
    "TP10_SL10", "TP10_SL15", "TP15_SL10", "TP15_SL15", "TP20_SL10", "TP20_SL15", 
    "TP10_SL20", "TP15_SL20", "TP20_SL20", "TP20_SL25", "TP25_SL20", "TP25_SL25",
    "TP25_SL30", "TP30_SL25", "TP20_SL30", "TP30_SL20", "TP30_SL30", "TP30_SL35",
    "TP35_SL30", "TP35_SL35", "TP35_SL40", "TP40_SL35", "TP30_SL40", "TP40_SL30",
    "TP40_SL40"
]


# Paths
IPositionExitCombinationPath = Literal[
    "01_TP10_SL10", "02_TP10_SL15", "03_TP15_SL10", "04_TP15_SL15", "05_TP20_SL10", "06_TP20_SL15", 
    "07_TP10_SL20", "08_TP15_SL20", "09_TP20_SL20", "10_TP20_SL25", "11_TP25_SL20", "12_TP25_SL25",
    "13_TP25_SL30", "14_TP30_SL25", "15_TP20_SL30", "16_TP30_SL20", "17_TP30_SL30", "18_TP30_SL35",
    "19_TP35_SL30", "20_TP35_SL35", "21_TP35_SL40", "22_TP40_SL35", "23_TP30_SL40", "24_TP40_SL30",
    "25_TP40_SL40"
]


# Position Exit Combination Record
class IPositionExitCombinationRecord(TypedDict):
    take_profit: float
    stop_loss: float
    path: IPositionExitCombinationPath


# Database containing all combination records
IPositionExitCombinationDatabase = Dict[IPositionExitCombinationID, IPositionExitCombinationRecord]









## Configuration ##




# Epoch Config
# The configuration to be used by the Epoch and influence the entire infrastructure.
class IEpochConfig(TypedDict):
    # Random seed to be set on all required libraries in order to guarantee reproducibility
    seed: int

    # Identifier, must be preffixed with "_". For example: "_EPOCHNAME"
    id: str

    # The range of the Epoch
    start: int
    end: int

    # The range that will be used to train the KerasModels
    training_start: int
    training_end: int

    # The identifier of the classification training data for unit tests
    ut_class_training_data_id: Union[str, None]

    # The Position Exit Combination that came victorious in the Regression Selection Process
    take_profit: Union[float, None]
    stop_loss: Union[float, None]






