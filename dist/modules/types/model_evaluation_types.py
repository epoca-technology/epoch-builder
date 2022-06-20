from typing import TypedDict, List





# Model Evaluation
# Evaluation performed right after the model is trained in order to get an overview of the
# potential accuracy, as well as the prediction type distribution.
# The evaluation works similarly to the backtests and it is only performed on the test
# dataset to ensure the model has not yet seen it.
class IModelEvaluation(TypedDict):
    # The number of evaluations performed on the Regression
    evaluations: int
    max_evaluations: int

    # The number of times the Regression predicted a price increase
    increase_num: int
    increase_successful_num: int

    # The number of times the Regression predicted a price decrease
    decrease_num: int
    decrease_successful_num: int

    # Accuracy
    increase_acc: int
    decrease_acc: int
    acc: int

    # Increase Predictions Overview
    increase_list: List[float]
    increase_max: float
    increase_min: float
    increase_mean: float
    increase_successful_max: float
    increase_successful_min: float
    increase_successful_mean: float

    # Decrease Predictions Overview
    decrease_list: List[float]
    decrease_max: float
    decrease_min: float
    decrease_mean: float
    decrease_successful_max: float
    decrease_successful_min: float
    decrease_successful_mean: float

    # Outcomes
    increase_outcomes: int
    decrease_outcomes: int