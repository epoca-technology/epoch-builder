from modules.keras_models import IKerasModelConfig




def validate(
    config: IKerasModelConfig,
    name: str,
    required_units: int = 0,
    required_dropout_rates: int = 0,
) -> None:
    """Given a configuration dict and a series of requirements, it will validate the 
    integrity of it.

    Raises:
        ValueError:
            If the provided configuration doesnt match the model's requirements.
    """
    # Make sure the name matches
    if config['name'] != name:
        raise ValueError(f"Model Name Missmatch. {config['name']} != {name}")

    # Make sure the predictions have been provided
    if not isinstance(config['predictions'], int):
        raise ValueError(f"The provided predictions is not a valid integer. Received {config['predictions']}")

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