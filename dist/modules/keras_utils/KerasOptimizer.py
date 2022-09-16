from typing import Union
from keras.optimizers import Adam, RMSprop
from keras.optimizers.schedules.learning_rate_schedule import InverseTimeDecay
from modules._types import IKerasOptimizer, IKerasTrainingConfig


# Optimizer Instance Type
IKerasOptimizerInstance = Union[Adam, RMSprop]



# Optimizer Instance Factory
def KerasOptimizer(func_name: IKerasOptimizer, learning_rate: float, config: IKerasTrainingConfig) -> IKerasOptimizerInstance:
    """Returns the instance of the optimizer function based on the provided
    function name and learning rate.

    Args:
        func_name: IKerasMetric
            The name of the loss function to be initialized
        learning_rate: float
            The learning rate that will be used to train the model.
        config: IKerasTrainingConfig
            The configuration to be used by the training instance.

    Returns:
        IKerasOptimizerInstance
    
    Raises:
        ValueError:
            If the function name does not match any function in the conditionings.
    """
    # Initialize the learning rate schedule
    lr_schedule: Union[InverseTimeDecay, float] = LearningRateSchedule(learning_rate, config)

    # Initialize the appropiate optimizer
    if func_name == "adam":
        return Adam(lr_schedule)
    elif func_name == "rmsprop":
        return RMSprop(lr_schedule)
    else:
        raise ValueError(f"The optimizer function for {func_name} was not found.")






# Learning Rate Schedule
def LearningRateSchedule(learning_rate: float, config: IKerasTrainingConfig) -> Union[InverseTimeDecay, float]:
    """Builds the learning rate schedule based on the provided learning rate. If the value
    is equals to -1, it will return an InverseTimeDecay Instance. Otherwise, it will just
    return the provided value.

    Args:
        learning_rate: float
            The learning rate to be used.
        config: IKerasTrainingConfig
            The keras training config in case the InverseTimeDecay is used.

    Returns:
        Union[InverseTimeDecay, float]
    """
    if learning_rate == -1:
        return InverseTimeDecay(
            initial_learning_rate=config["initial_lr"],
            decay_steps=config["decay_steps"],
            decay_rate=config["decay_rate"],
            staircase=False,
            name="LearningRateInverseTimeDecay"
        )
    else:
        return learning_rate