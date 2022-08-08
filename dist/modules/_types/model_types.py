from typing import TypedDict, List, Union, Dict, Literal
from pandas import DataFrame
from modules._types.discovery_types import IDiscovery
from modules._types.prediction_types import IPrediction
from modules._types.interpreter_types import IPercentChangeInterpreterConfig, IProbabilityInterpreterConfig, \
    IConsensusInterpreterConfig
from modules._types.keras_models_types import IKerasModelSummary
from modules._types.xgb_models_types import IXGBModelSummary








## Model Types ##


# Types of models supported by the project
IModelType = Literal[
    "KerasRegressionModel",         # KR_
    "KerasClassificationModel",     # KC_
    "XGBRegressionModel",           # XGBR_
    "XGBClassificationModel",       # XGBC_
    "ConsensusModel"                # CON_
]


# Trainable Model Types
ITrainableModelType = Literal[
    "keras_regression",     # KerasRegressionModel
    "keras_classification", # KerasClassificationModel
    "xgb_regression",       # XGBRegressionModel
    "xgb_classification"    # XGBClassificationModel
]


# Model ID Prefix
IModelIDPrefix = Literal[
    "KR_",     # KerasRegressionModel
    "KC_",     # KerasClassificationModel
    "XGBR_",   # XGBRegressionModel
    "XGBC_",   # XGBClassificationModel
    "CON_",    # ConsensusModel
]


# ModelType's Functions accept identifiers or prefixes.
IPrefixOrID = Union[IModelIDPrefix, str]


# The extension of trained models
ITrainableModelExtension = Literal["h5", "json"]










## Regression Configurations ##


# Regresion Configuration
# The general configuration that should be followed by regressions.
class IRegressionConfig(TypedDict):
    # The identifier of the model
    id: str

    # Important information regarding the model
    description: str

    # Regression Model Type
    # Default: will generate all predictions in one go.
    # Autoregressive: will generate 1 prediction at a time and feed it to itself as an input 
    autoregressive: bool

    # The number of candlesticks it will lookback to make a prediction
    lookback: int

    # The number of predictions it will generate
    predictions: int

    # The discovery performed prior to saving the model
    discovery: IDiscovery



# Keras Regresion Configuration
# The configuration that was used to train and will predict based on.
class IKerasRegressionConfig(IRegressionConfig):
    # The summary of the KerasModel
    summary: IKerasModelSummary


# XGBoost Regresion Configuration
# The configuration that was used to train and will predict based on.
class IXGBRegressionConfig(IRegressionConfig):
    # The summary of the XGBoostModel
    summary: IXGBModelSummary







## Classification Configurations ##


# Classification Configuration
# The general configuration that should be followed by classifications.
class IClassificationConfig(TypedDict):
    # The identifier of the model
    id: str

    # Important information regarding the trained model
    description: str

    # The identifier of the training data used
    training_data_id: str

    # The list of KerasRegressionModel|XGBRegressionModel attached to the Classification
    models: List[Dict] # IModel does not exist yet

    # Optional Technical Analysis Features
    include_rsi: bool       # Momentum
    include_aroon: bool     # Trend

    # The total number of features that will be used by the model to predict
    features_num: int

    # The percentage the price needs to change in order to be considered up or down
    price_change_requirement: float

    # The discovery performed prior to saving the model
    discovery: IDiscovery



# Keras Classification Configuration
# The configuration that was used to train and will predict based on.
class IKerasClassificationConfig(IClassificationConfig):
    # The summary of the KerasModel
    summary: IKerasModelSummary




# XGBoost Classification Configuration
# The configuration that was used to train and will predict based on.
class IXGBClassificationConfig(IClassificationConfig):
    # The summary of the XGBoostModel
    summary: IXGBModelSummary












## RegressionModel Configurations ##



# RegressionModel Configuration
# The general configuration that should be followed by regression models.
class IRegressionModelConfig(TypedDict):
    # The ID of the saved regression model
    regression_id: str

    # The interpreter that will determine the prediction's result. This value is only present
    # when the function get_model is used.
    interpreter: Union[IPercentChangeInterpreterConfig, None]



# KerasRegressionModel Configuration
# The configuration used by the Keras Regression. Only exists when the model is initialized.
class IKerasRegressionModelConfig(IRegressionModelConfig):
    regression: Union[IKerasRegressionConfig, None]


# XGBRegressionModel Configuration
# The configuration used by the XGBoost Regression. Only exists when the model is initialized.
class IXGBRegressionModelConfig(IRegressionModelConfig):
    regression: Union[IXGBRegressionConfig, None]







## ClassificationModel Configurations ##


# ClassificationModel Configuration
# The general configuration that should be followed by classification models.
class IClassificationModelConfig(TypedDict):
    # The ID of the saved keras classification model
    classification_id: str

    # The interpreter that will determine the prediction's result. This value is only present
    # when the function get_model is used.
    interpreter: Union[IProbabilityInterpreterConfig, None]




# KerasClassificationModel Configuration
# The configuration used by the Keras Classification. Only exists when the model is initialized.
class IKerasClassificationModelConfig(IClassificationModelConfig):
    classification: Union[IKerasClassificationConfig, None]



# XGBClassificationModel Configuration
# The configuration used by the XGBoost Classification. Only exists when the model is initialized.
class IXGBClassificationModelConfig(IClassificationModelConfig):
    classification: Union[IXGBClassificationConfig, None]






# ConsensusModel Configuration
# The configuration that will be use to generate and interpret predictions.
class IConsensusModelConfig(TypedDict):
    # The list of KerasRegressionModel|KerasClassificationModel|XGBRegressionModel|
    # XGBClassificationModel attached to the ConsensusModel. This value is only populated 
    # when get_model is invoked
    sub_models: Union[List[Dict], None] # IModel does not exist yet.

    # The interpreter that will determine the prediction's result. Must represent at least 51%
    # of the provided sub models.
    interpreter: IConsensusInterpreterConfig










## Model ##



# Model
# The final state of a KerasRegressionModel|KerasClassificationModel|XGBRegressionModel|
# XGBClassificationModel|ConsensusModel once an 
# instance is initialized.
# The type of a model can be determined based on its configuration. Existing models are:
# 1) KerasRegressionModel: a model with a single keras_regression.
# 2) XGBRegressionModel: a model with a single xgb_regression.
# 3) KerasClassificationModel: a model with a single keras_classification.
# 4) XGBClassificationModel: a model with a single xgb_classification.
# 5) ConsensusModel: a model with any number of keras_classification|xgb_classification.
class IModel(TypedDict):
    # Identity of the Model. This value will be used by the cache system when storing/retrieving predictions.
    id: str

    # KerasRegression Configurations
    keras_regressions: Union[List[IKerasRegressionModelConfig], None]

    # XGBRegression Configurations
    xgb_regressions: Union[List[IXGBRegressionModelConfig], None]

    # KerasClassification Configurations
    keras_classifications: Union[List[IKerasClassificationModelConfig], None]

    # XGBClassification Configurations
    xgb_classifications: Union[List[IXGBClassificationModelConfig], None]

    # ConsensusModel Configuration
    consensus: Union[IConsensusModelConfig, None]











# Model Interface
# KerasClassificationModel, XGBClassificationModel and ConsensusModel implement the following interface
# in order to ensure compatibility across any of the processes.
class ModelInterface:
    # Init
    def __init__(self, config: IModel, enable_cache: bool = False):
        raise NotImplementedError("Model.__init__ has not been implemented.")

    # Performs a prediction based on the current time
    def predict(self, current_timestamp: int, lookback_df: Union[DataFrame, None] = None) -> IPrediction:
        raise NotImplementedError("Model.predict has not been implemented.")

    # Retrieves the lookback set on the Model
    def get_lookback(self) -> int:
        raise NotImplementedError("Model.get_lookback has not been implemented.")

    # Retrieves the configuration of the Model after being initialized
    def get_model(self) -> IModel:
        raise NotImplementedError("Model.get_model has not been implemented.")

    # Checks if a config is for the Model
    @staticmethod
    def is_config(model: IModel) -> bool:
        raise NotImplementedError("Model.is_config has not been implemented.")





# Regression Model Interface
# KerasRegressionModel and XGBRegressionModel implement the following interface
# in order to ensure compatibility across any of the processes.
class RegressionModelInterface(ModelInterface):
    # Generates a feature based on the current time
    def feature(self, current_timestamp: int, lookback_df: Union[DataFrame, None] = None) -> float:
        raise NotImplementedError("RegressionModel.feature has not been implemented.")












