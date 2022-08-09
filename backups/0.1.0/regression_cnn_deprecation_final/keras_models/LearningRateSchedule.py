from keras.optimizer_v2.learning_rate_schedule import InverseTimeDecay





def LearningRateSchedule(initial_learning_rate: float, decay_steps: float, decay_rate: float) -> InverseTimeDecay:
    """Builds an inverse time decay to be used the learning rate for training.

    Args:
        initial_learning_rate: float
            A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
        decay_steps: float
            How often to apply decay.
        decay_rate: float
            A Python number. The decay rate for the learning rate per step.

    Returns:
        InverseTimeDecay
    """
    return InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
        name="LearningRateInverseTimeDecay"
    )