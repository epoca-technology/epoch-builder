from keras import Sequential
from modules.keras_models import IKerasModelConfig
import modules.keras_models.NeuralNetworks.RegressionNeuralNetworks as KerasRegressionModels
import modules.keras_models.NeuralNetworks.ClassificationNeuralNetworks as KerasClassificationModels





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
            If the type if invalid.
            if the model configuration is invalid
    """
    # Build a Regression Model
    if config["name"][0:2] == "R_":
        return getattr(KerasRegressionModels, config["name"])(config)
    elif config["name"][0:2] == "C_":
        return getattr(KerasClassificationModels, config["name"])(config)
    else:
        raise ValueError(f"The Keras Model Type could not be found for {config['name']}.")