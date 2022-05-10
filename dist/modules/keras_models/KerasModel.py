from typing import Union, Any
from keras import Sequential
from modules.keras_models import IKerasModelConfig
import modules.keras_models.KerasRegressionModels as KerasRegressionModels
import modules.keras_models.KerasClassificationModels as KerasClassificationModels





def KerasModel(model_type: str, config: IKerasModelConfig) -> Union[Sequential, Any]:
    """Based on a given configuration it returns the instance of a Keras Model that
    is ready to be trained.

    Args:
        model_type: str
            The type of the model that will be returned. It can be 'regression'|'classification'
        config: IKerasModelConfig
            The Keras Model Configuration.
    Returns:
        ...
    
    Raises:
        ValueError:
            If the type if invalid.
            if the model configuration is invalid
    """
    # Build a Regression Model
    if model_type == 'regression':
        return getattr(KerasRegressionModels, config["name"])(config)
    elif model_type == 'classification':
        return getattr(KerasClassificationModels, config["name"])(config)
    else:
        raise ValueError(f"The Keras Model Type {model_type} is invalid.")