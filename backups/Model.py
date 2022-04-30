from typing import Union
from modules.model import SingleModel, MultiModel, IModel





def Model(config: IModel) -> Union[SingleModel, MultiModel]:
    """Returns the instance of a SingleModel or a MultiModel based on the 
    provided configuration.

    Args:
        config: IModel
            The configuration of module to return the instance of.

    Returns:
        Union[SingleModel, MultiModel]
    """
    return MultiModel(config) if len(config['single_models']) > 1 else SingleModel(config)