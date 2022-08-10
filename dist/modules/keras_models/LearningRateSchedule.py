from typing import Union
from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay





def LearningRateSchedule(
    learning_rate: float,
    initial_learning_rate: float, 
    decay_steps: float, 
    decay_rate: float
) -> Union[InverseTimeDecay, float]:
    """Builds an inverse time decay to be used the learning rate for training.

    Args:
        learning_rate: float
            The learning rate to be used. If the value is equals to -1 it will return
            the InverseTimeDecay instance. Otherwise, it will just return the provided
            learning rate.
        initial_learning_rate: float
            A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
        decay_steps: float
            How often to apply decay.
        decay_rate: float
            A Python number. The decay rate for the learning rate per step.

    Returns:
        InverseTimeDecay
    """
    if learning_rate == -1:
        return InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
            name="LearningRateInverseTimeDecay"
        )
    else:
        return learning_rate