from keras import Sequential
from modules._types import IKerasModelConfig
import modules.keras_utils.NetworkArchitectureTemplates as NetworkArchitectureTemplates





def KerasModel(config: IKerasModelConfig) -> Sequential:
    """Based on a given configuration it returns the instance of a Keras Model that
    is ready to be trained.

    Args:
        config: IKerasModelConfig
            The Keras Model Configuration.
    Returns:
        Sequential
    
    Raises:
        ValueError:
            If the model's template cannot be found.
            if the model configuration is invalid
    """
    return getattr(NetworkArchitectureTemplates, config["name"])(config)