from typing import Union, List
from modules._types import IKerasModelConfig




def validate(
    config: IKerasModelConfig,
    name: str,
    units: int = 0,
    activations: int = 0,
    filters: int = 0,
    kernel_sizes: int = 0,
    pool_sizes: int = 0,
) -> None:
    """Given a configuration dict and a series of requirements, it will validate the 
    integrity of it.

    Args:
        config: IKerasModelConfig
            The configuration of the model that will be validated.
        name: str
            The name of the model that will be built.
        units: int
            The number of unit items that should have been provided in the config.
        activations: int
            The number of activation items that should have been provided in the config.
        filters: int
            The number of filter items  that should have been provided in the config.
        kernel_sizes: int
            The number of kernel_size items  that should have been provided in the config.
        pool_sizes: int
            The number of pool_size items  that should have been provided in the config.

    Raises:
        ValueError:
            If it did not receive correct lookback or prediction args.
            If the provided configuration doesnt match the model's requirements in any way.
    """
    # Make sure the name matches
    if config["name"] != name:
        raise ValueError(f"Model Name Missmatch. {config['name']} != {name}")

    # Make sure the autoregressive, lookback and predictions have been provided in case of a regression
    if not isinstance(config.get("lookback"), int) or not isinstance(config.get("predictions"), int):
        print(config.get("lookback"))
        print(config.get("predictions"))
        raise ValueError(f"The provided regression config (lookback and/or predictions) are invalid.")

    # Validate the units
    _validate_config_param(units, "units", config.get("units"))

    # Validate the required activations
    _validate_config_param(activations, "activations", config.get("activations"))

    # Validate the required filters
    _validate_config_param(filters, "filters", config.get("filters"))

    # Validate the required kernel_sizes
    _validate_config_param(kernel_sizes, "kernel_sizes", config.get("kernel_sizes"))

    # Validate the required pool_sizes
    _validate_config_param(pool_sizes, "pool_sizes", config.get("pool_sizes"))






def _validate_config_param(requirement: int, param_name: str, param: Union[List[Union[int, float, str]], None]) -> None:
    """Validates the configuration against the requirements. Note that this is a strict validation. As well as
    checking the requirements, it will make sure that no incorrect config params are allowed.

    Args:
        requirement: int
            The number of items required.
        param_name: str
            The name of the parameter to be validated.
        param: Union[List[Union[int, float, str]], None]
            The value of the parameter to be validated.

    Raises:
        ValueError:
            If the value is not required but provided by mistake.
            If the value is required and it wasn't provided.
            If the provided value does not meet the requirement.
    """
    if requirement == 0 and param is not None:
        raise ValueError(f"If there are no required {param_name} it should not be in the config. \
            Received {str(param)}")
    elif requirement > 0 and (param is None or len(param) < requirement):
        raise ValueError(f"The provided {param_name} did not meet the requirements. Received {str(param)}")