from typing import Union
from modules._types import IPrefixOrID, IModelType
from modules.model.ModelType import get_model_type
from modules.model.KerasRegressionModel import KerasRegressionModel
from modules.model.XGBRegressionModel import XGBRegressionModel
from modules.model.KerasClassificationModel import KerasClassificationModel
from modules.model.XGBClassificationModel import XGBClassificationModel


# Types of Sub Models
SubModelInstance = Union[KerasRegressionModel, XGBRegressionModel, KerasClassificationModel, XGBClassificationModel]



# Sub Model Factory
# Based on given configuration, it returns the appropiate Model Instance
def SubModelFactory(id: IPrefixOrID, enable_cache: bool = False) -> SubModelInstance:
    """Returns the instance of a Sub Model based on the provided configuration.

    Args:
        config: IModel
            The configuration of model to return the instance of.
        enable_cache: bool
            The state of the cache. If False, the model won't interact with the db.

    Returns:
        SubModelInstance
    """
    # Retrieve the model type
    model_type: IModelType = get_model_type(id)

    # Check if it is a Keras Regression Model
    if model_type == "KerasRegressionModel":
        return KerasRegressionModel(
            { "id": id, "keras_regressions": [{ "regression_id": id }] }, 
            enable_cache=enable_cache
        )

    # Check if it is a XGB Regression Model
    elif model_type == "XGBRegressionModel":
        return XGBRegressionModel(
            { "id": id, "xgb_regressions": [{ "regression_id": id }] }, 
            enable_cache=enable_cache
        )

    # Check if it is a Keras Classification Model
    elif model_type == "KerasClassificationModel":
        return KerasClassificationModel(
            { "id": id, "keras_classifications": [{ "classification_id": id }] }, 
            enable_cache=enable_cache
        )

    # Check if it is a XGB Classification Model
    elif model_type == "XGBClassificationModel":
        return XGBClassificationModel(
            { "id": id, "xgb_classifications": [{ "classification_id": id }] }, 
            enable_cache=enable_cache
        )

    # Otherwise, the provided configuration is invalid
    else:
        print(id)
        raise ValueError("Couldnt find an instance for the provided model configuration.")