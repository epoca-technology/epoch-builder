from typing import Union
from keras import Sequential
from modules.keras_models import IKerasModelConfig
import modules.keras_models.KerasRegressionLSTM as KerasRegressionLSTM



def KerasRegressionModel(config: IKerasModelConfig) -> Sequential:
    """Based on a given configuration it returns the instance of a Keras Model that
    is ready to be trained.

    Args:
        config: IKerasModelConfig
            The Keras Model Configuration.

    Returns:
        Sequential
    
    Raises:
        ValueError:
            If the keras model's name is not found.
    """
    ## Standard Models ##


    # Dense
    # @TODO

    # Convolutional
    # @TODO


    # Long Short Term Memory
    if config["name"] == "LSTM_STACKLESS":
        return KerasRegressionLSTM.LSTM_STACKLESS(config)
    elif config["name"] == "LSTM_SOFT_STACK_BALANCED_DROPOUT":
        return KerasRegressionLSTM.LSTM_SOFT_STACK_BALANCED_DROPOUT(config)
    elif config["name"] == "LSTM_SOFT_STACK":
        return KerasRegressionLSTM.LSTM_SOFT_STACK(config)
    elif config["name"] == "LSTM_MEDIUM_STACK_BALANCED_DROPOUT":
        return KerasRegressionLSTM.LSTM_MEDIUM_STACK_BALANCED_DROPOUT(config)
    elif config["name"] == "LSTM_MEDIUM_STACK":
        return KerasRegressionLSTM.LSTM_MEDIUM_STACK(config)
    elif config["name"] == "LSTM_HARD_STACK_BALANCED_DROPOUT":
        return KerasRegressionLSTM.LSTM_HARD_STACK_BALANCED_DROPOUT(config)
    elif config["name"] == "LSTM_HARD_STACK":
        return KerasRegressionLSTM.LSTM_HARD_STACK(config)


    ## AutoRegressive Models ##


    # Dense
    # @TODO


    # Convolutional
    # @TODO


    # Long Short Term Memory
    # @TODO

    
    # Model not found
    else:
        raise ValueError(f"The Keras Model {config['name']} was not found.")