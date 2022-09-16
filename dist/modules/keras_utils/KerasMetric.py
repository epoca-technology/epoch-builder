from typing import Union
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from modules._types import IKerasMetric


# Metric Instance Type
IKerasMetricInstance = Union[MeanAbsoluteError, MeanSquaredError]



# Metric Instance Factory
def KerasMetric(func_name: IKerasMetric) -> IKerasMetricInstance:
    """Returns the instance of the metric function based on the provided
    function name.

    Args:
        func_name: IKerasMetric
            The name of the loss function to be initialized
    Returns:
        IKerasMetricInstance
    
    Raises:
        ValueError:
            If the function name does not match any function in the conditionings.
    """
    if func_name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif func_name == "mean_squared_error":
        return MeanSquaredError()
    else:
        raise ValueError(f"The metric function for {func_name} was not found.")