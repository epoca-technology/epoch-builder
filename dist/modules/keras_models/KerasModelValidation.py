from ..keras_models.types import IKerasModelConfig


def validate(
    config: IKerasModelConfig,
    model_type: str,
    name: str,
    required_units: int = 0,
    required_dropout_rates: int = 0,
    required_activations: int = 0,
    required_filters: int = 0,
    required_pool_sizes: int = 0,
) -> None:
    """Given a configuration dict and a series of requirements, it will validate the 
    integrity of it.

    Args:
        config: IKerasModelConfig
            The configuration of the model that will be validated.
        model_type: str
            The type of model that will be validated.
        name: str
            The name of the model that will be built.
        required_units: int
            The number of unit items that should have been provided in the config.
        required_dropout_rates: int
            The number of dropout rates items that should have been provided in the config.
        required_activations: int
            The number of activation items that should have been provided in the config.
        required_filters: int
            The number of filter items  that should have been provided in the config.
        required_pool_sizes: int
            The number of pool_size items  that should have been provided in the config.

    Raises:
        ValueError:
            If the provided configuration doesnt match the model's requirements.
    """
    # Make sure the name matches
    if config['name'] != name:
        raise ValueError(f"Model Name Missmatch. {config['name']} != {name}")

    # Make sure the predictions have been provided
    if model_type == 'regression' and (not isinstance(config['lookback'], int) or not isinstance(config['predictions'], int)):
        raise ValueError(f"The provided lookback and or predictions are not valid integers. Received: {config['lookback']}, {config['predictions']}")

    # Validate the units
    if required_units == 0 and config.get('units') is not None:
        raise ValueError(f"If there are no required units there shouldnt be units in the config. Received {str(config['units'])}")
    elif required_units > 0 and (config.get('units') is None or len(config['units']) < required_units):
        raise ValueError(f"The provided units did not meet the requirements. Received {str(config['units'])}")

    # Validate the required dropout rates
    if required_dropout_rates == 0 and config.get('dropout_rates') is not None:
        raise ValueError(f"If there are no required dropout_rates there shouldnt be dropout_rates in the config. \
            Received {str(config['dropout_rates'])}")
    elif required_dropout_rates > 0 and (config.get('dropout_rates') is None or len(config['dropout_rates']) < required_dropout_rates):
        raise ValueError(f"The provided dropout_rates did not meet the requirements. Received {str(config['dropout_rates'])}")

    # Validate the required activations
    if required_activations == 0 and config.get('activations') is not None:
        raise ValueError(f"If there are no required activations there shouldnt be activations in the config. \
            Received {str(config['activations'])}")
    elif required_activations > 0 and (config.get('activations') is None or len(config['activations']) < required_activations):
        raise ValueError(f"The provided activations did not meet the requirements. Received {str(config['activations'])}")

    # Validate the required filters
    if required_filters == 0 and config.get('filters') is not None:
        raise ValueError(f"If there are no required filters there shouldnt be filters in the config. \
            Received {str(config['filters'])}")
    elif required_filters > 0 and (config.get('filters') is None or len(config['filters']) < required_filters):
        raise ValueError(f"The provided filters did not meet the requirements. Received {str(config['filters'])}")

    # Validate the required pool_sizes
    if required_pool_sizes == 0 and config.get('pool_sizes') is not None:
        raise ValueError(f"If there are no required pool_sizes there shouldnt be pool_sizes in the config. \
            Received {str(config['pool_sizes'])}")
    elif required_pool_sizes > 0 and (config.get('pool_sizes') is None or len(config['pool_sizes']) < required_pool_sizes):
        raise ValueError(f"The provided pool_sizes did not meet the requirements. Received {str(config['pool_sizes'])}")