from typing import TypedDict, List
from modules._types.discovery_types import IDiscoveryPayload
from modules._types.model_evaluation_types import IModelEvaluation
from modules._types.model_types import IModel





# Selected Regression
# All the insights regarding the selected RegressionModel.
class ISelectedRegression(TypedDict):
    # The identifier of the model
    id: str

    # The selected RegressionModel
    model: IModel

    # The discovery of the Regression
    discovery: IDiscoveryPayload

    # The evaluation of the Regression
    evaluation: IModelEvaluation

    # The history of the points median. It should be a list of 10 items representing
    # 10%,20%,30%,40%,50%,60%,70%,80%,90% and 100% of the points_hist
    points_median_hist: List[float]





# Regression Selection File
# The result of a RegressionSelection that contains all the information related
# to the selection and the models in it.
class IRegressionSelectionFile(TypedDict):
    # Universally Unique Identifier (uuid4)
    id: str

    # The date in which the regression selection was created
    creation: int
    
    # The mean of all the discoveries successful_mean
    price_change_mean: float

    # The list of selected regressions
    selection: List[ISelectedRegression]



