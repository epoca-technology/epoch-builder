from typing import TypedDict, Union, List, Literal, Dict
from modules._types.model_types import ITrainableModelType
from modules._types.keras_models_types import IKerasLoss, IKerasMetric, IKerasModelConfig




#############
## General ##
#############


IHyperparamsCategory = Literal[
    # General
    "UNIT_TEST",

    # Keras
    "DNN", "CNN", "LSTM", "CLSTM",

    # XGBoost
    # ...
]






## KERAS TYPES ##



# Neural Networks
# In order to find the models that best fit to the dataset, the hyperparams module generates
# many variations on several kinds of neural networks.


# Name of the variations within each network
IKerasHyperparamsNetworkVariationName = Literal[
    # Keras Regression Series
    "KR_DNN_S2", "KR_DNN_S3", "KR_DNN_S4", "KR_DNN_S5",
    "KR_CNN_S2", "KR_CNN_S2_MP", "KR_CNN_S3", "KR_CNN_S3_MP", "KR_CNN_S4", "KR_CNN_S4_MP", "KR_CNN_S5", "KR_CNN_S5_MP",
    "KR_LSTM_S2", "KR_LSTM_S3", "KR_LSTM_S4", "KR_LSTM_S5",
    "KR_CLSTM_S2", "KR_CLSTM_S2_MP", "KR_CLSTM_S3", "KR_CLSTM_S3_MP", "KR_CLSTM_S4", "KR_CLSTM_S4_MP", "KR_CLSTM_S5", "KR_CLSTM_S5_MP",

    # Keras Classification Series
    "KC_DNN_S2", "KC_DNN_S3", "KC_DNN_S4", "KC_DNN_S5",
    "KC_CNN_S2", "KC_CNN_S2_MP", "KC_CNN_S3", "KC_CNN_S3_MP", "KC_CNN_S4", "KC_CNN_S4_MP", "KC_CNN_S5", "KC_CNN_S5_MP",
    "KC_LSTM_S2", "KC_LSTM_S3", "KC_LSTM_S4", "KC_LSTM_S5",
    "KC_CLSTM_S2", "KC_CLSTM_S2_MP", "KC_CLSTM_S3", "KC_CLSTM_S3_MP", "KC_CLSTM_S4", "KC_CLSTM_S4_MP", "KC_CLSTM_S5", "KC_CLSTM_S5_MP"
]


# Variations per network
IKerasHyperparamsNetworkVariations = Dict[IKerasHyperparamsNetworkVariationName, List[IKerasModelConfig]]


# Variations by network
class IKerasHyperparamsNetworks(TypedDict):
    DNN: IKerasHyperparamsNetworkVariations
    CNN: IKerasHyperparamsNetworkVariations
    LSTM: IKerasHyperparamsNetworkVariations
    CLSTM: IKerasHyperparamsNetworkVariations










# Keras Loss
# The loss and metric to use to train models.
class IKerasHyperparamsLoss(TypedDict):
    name: IKerasLoss
    metric: IKerasMetric









# Keras Hyperparams Receipt
# Once all configs have been saved, a receipt is generated in order to
# summarize the models that will be trained in hyperparams mode.


# Network Receipt
class IKerasHyperparamsNetworkReceipt(TypedDict):
    name: str
    models: int
    batches: int



# Main Receipt
class IKerasHyperparamsReceipt(TypedDict):
    creation: str
    model_type: ITrainableModelType
    batch_size: int
    training_data_id: Union[str, None]
    output_name: str
    total_models: int
    networks: List[IKerasHyperparamsNetworkReceipt]



