from typing import TypedDict, List
from modules.types.model_types import IModel







# Model Result
# The Selection Result for a single model. 
class IModelResult(TypedDict):
    # The model's dict
    model: IModel

    # The median of all the accumulated points within the combination
    points_median: float

    # The history of the points median. It should be a list of 10 items representing
    # 10%,20%,30%,40%,50%,60%,70%,80%,90% and 100% of the points_hist
    points_median_hist: List[float]

    # Positions
    long_num: int     # Number of closed long positions
    short_num: int    # Number of closed short positions

    # Accuracy
    long_acc: float     # Longs Accuracy
    short_acc: float    # Shorts Accuracy
    general_acc: float  # General Accuracy







# Combination Result
# The result issued by a TakeProfit/StopLoss Combination.
class ICombinationResult(TypedDict):
    # Combination
    combination_id: str

    # The total number of models within the combination
    models_num: int

    # The mean of the selected models' points medians
    points_mean: float

    # The list of ordered model results
    model_results: List[IModelResult]








# Regression Selection File
# The result of a RegressionSelection that contains all the information related
# to the selection and the models in it.
class IRegressionSelectionFile(TypedDict):
    # Universally Unique Identifier (uuid4)
    id: str
    
    # The number of models that were selected based on their points medians
    models_limit: int

    # The data range covered in the backtest results
    start: int
    end: int

    # The number of models that were put through the selection
    models_num: int

    # The list of ordered combination results 
    results: List[ICombinationResult]



