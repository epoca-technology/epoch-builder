from typing import Union
from keras.losses import MeanAbsoluteError, MeanSquaredError
from modules._types import IKerasLoss


# Loss Instance Type
IKerasLossInstance = Union[MeanAbsoluteError, MeanSquaredError]



# Loss Instance Factory
def KerasLoss(func_name: IKerasLoss) -> IKerasLossInstance:
    """Returns the instance of the loss function based on the provided
    function name.

    Args:
        func_name: IKerasLoss
            The name of the loss function to be initialized
    Returns:
        IKerasLossInstance
    
    Raises:
        ValueError:
            If the function name does not match any function in the conditionings.
    """
    if func_name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif func_name == "mean_squared_error":
        return MeanSquaredError()
    else:
        raise ValueError(f"The loss function for {func_name} was not found.")