from typing import TypedDict, Union, List, Literal
from modules.types.model_types import ITrainableModelType
from modules.types.keras_models_types import IKerasLoss, IKerasClassificationMetric



## KERAS TYPES ##


# Keras Loss
# The loss and metric to use to train models.
class IKerasHyperparamsLoss(TypedDict):
    name: IKerasLoss
    metric: Union[IKerasClassificationMetric, None] # Metric is only populated in Classification Models





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



