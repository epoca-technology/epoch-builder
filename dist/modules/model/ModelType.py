from typing import List
from modules.types import IModelType, ITrainableModelType, IModelIDPrefix, IPrefixOrID



## Constants ##


# Model Types
MODEL_TYPES: List[IModelType] = [
    "ArimaModel", "RegressionModel", "ClassificationModel", "XGBRegressionModel", 
    "XGBClassificationModel", "ConsensusModel"
]


# Trainable Model Types
TRAINABLE_MODEL_TYPES: List[ITrainableModelType] = [
    "keras_regression", "keras_classification", "xgb_regression", "xgb_classification"
]
TRAINABLE_REGRESSION_MODEL_TYPES: List[ITrainableModelType] = ["keras_regression", "xgb_regression"]
TRAINABLE_CLASSIFICATION_MODEL_TYPES: List[ITrainableModelType] = ["keras_classification", "xgb_classification"]


# Model ID Prefixes
MODEL_ID_PREFIXES: List[IModelIDPrefix] = ["A", "R_", "C_", "XGBR_", "XGBC_", "CON_"]





## Validators ## 




def validate_id(model_type: IModelType, id: str) -> None:
    """Validates and ID based on the model's type. If any requirement is
    not met, an error will be raised.

    Args:
        model_type: IModelType
            The type of model.
        id: str
            The ID to be given to the model.

    Raises:
        ValueError:
            If the Model ID is not at least 4 characters long
            If the provided model_type is invalid.
            If the prefix does not match any models.
            If the prefix does not match the provided model_type
    """
    # Make sure the model is at least 4 characters long
    if len(id) < 4:
        raise ValueError(f"The provided model ID {id} is invalid as it must contain at least 4 characters.")

    # Make sure the provided model_type is valid
    if model_type not in MODEL_TYPES:
        raise ValueError(f"The provided model type {model_type} is invalid.")

    # Retrieve the prefix
    prefix: IModelIDPrefix = get_prefix(id)

    # Validate an ArimaModel
    if model_type == "ArimaModel" and prefix != "A":
        raise ValueError(f"The ArimaModel ID contains an invalid prefix: {prefix}")

    # Validate a RegressionModel
    elif model_type == "RegressionModel" and prefix != "R_":
        raise ValueError(f"The RegressionModel ID contains an invalid prefix: {prefix}")

    # Validate a ClassificationModel
    elif model_type == "ClassificationModel" and prefix != "C_":
        raise ValueError(f"The ClassificationModel ID contains an invalid prefix: {prefix}")

    # Validate a XGBRegressionModel
    elif model_type == "XGBRegressionModel" and prefix != "XGBR_":
        raise ValueError(f"The XGBRegressionModel ID contains an invalid prefix: {prefix}")

    # Validate a XGBClassificationModel
    elif model_type == "XGBClassificationModel" and prefix != "XGBC_":
        raise ValueError(f"The XGBClassificationModel ID contains an invalid prefix: {prefix}")

    # Validate a ConsensusModel
    elif model_type == "ConsensusModel" and prefix != "CON_":
        raise ValueError(f"The ConsensusModel ID contains an invalid prefix: {prefix}")







## Helpers ##



def is_regression(model_id: IPrefixOrID) -> bool:
    """Checks if a given model id belongs to any of the supported
    regressions.

    Args:
        model_id: IPrefixOrID
            The id of the model.

    Returns:
        bool
    """
    # Retrieve the prefix
    prefix: IModelIDPrefix = get_prefix(model_id)

    # Check if it is a regression
    return prefix == "A" or prefix == "R_" or prefix == "XGBR_"










## Core Retrievers ##







def get_model_type(model_id: IPrefixOrID) -> IModelType:
    """Retrieves the model type based on provided ID.

    Args:
        model_id: IPrefixOrID
            The ID of the model.

    Returns:
        IModelType

    Raises:
        ValueError:
            If the id/prefix does not match any models.
    """
    # Retrieve the prefix
    prefix: IModelIDPrefix = get_prefix(model_id)

    # Return the model type based on the prefix
    if prefix == "A":
        return "ArimaModel"

    elif prefix == "R_":
        return "RegressionModel"

    elif prefix == "C_":
        return "ClassificationModel"

    elif prefix == "XGBR_":
        return "XGBRegressionModel"

    elif prefix == "XGBC_":
        return "XGBClassificationModel"

    else:
        return "ConsensusModel"







def get_trainable_model_type(model_id: IPrefixOrID) -> ITrainableModelType:
    """Retrieves the trainable model type based on provided ID.

    Args:
        model_id: IPrefixOrID
            The ID of the model.

    Returns:
        ITrainableModelType

    Raises:
        ValueError:
            If the id/prefix does not match any models.
            if the prefix does not match a trainable model.
    """
    # Retrieve the prefix
    prefix: IModelIDPrefix = get_prefix(model_id)

    # Check if it is a KerasRegression
    if prefix == "R_":
        return "keras_regression"

    # Check if it is a KerasClassification
    elif prefix == "C_":
        return "keras_classification"

    # Check if it is a XGBRegression
    elif prefix == "XGBR_":
        return "xgb_regression"

    # Check if it is a XGBClassification
    elif prefix == "XGBC_":
        return "xgb_classification"

    # Otherwise, the provided model id does not have a trainable model type
    else:
        raise ValueError(f"The provided Model ID does not have a trainable model type: {model_id}")





def get_prefix_by_trainable_model_type(model_type: ITrainableModelType) -> IModelIDPrefix:
    """Retrieves the prefix of a model based on its trainable type.

    Args:
        model_type: ITrainableModelType
            The trainable type of the model.

    Returns:
        IModelIDPrefix

    Raises:
        ValueError:
            If no prefix is found.
    """
    if model_type == "keras_regression":
        return "R_"
    elif model_type == "keras_classification":
        return "C_"
    elif model_type == "xgb_regression":
        return "XGBR_"
    elif model_type == "xgb_classification":
        return "XGBC_"
    else:
        raise ValueError(f"Could not find a prefix for: {model_type}")






def get_prefix(prefix_or_id: IPrefixOrID) -> IModelIDPrefix:
    """Returns the model's prefix based on a string.

    Args:
        prefix_or_id: IPrefixOrID
            The prefix or ID of the model.

    Returns:
        IModelIDPrefix

    Raises:
        ValueError:
            If the id/prefix does not match any models.
    """
    # Init the length of the prefix or id
    str_len: int = len(prefix_or_id)

    # Check if it is an ArimaModel
    if str_len >= 1 and prefix_or_id[0] == "A":
        return "A"

    # Check if it is a RegressionModel
    elif str_len >= 2 and prefix_or_id[0:2] == "R_":
        return "R_"

    # Check if it is a ClassificationModel
    elif str_len >= 2 and prefix_or_id[0:2] == "C_":
        return "C_"

    # Check if it is a XGBRegressionModel
    elif str_len >= 5 and prefix_or_id[0:5] == "XGBR_":
        return "XGBR_"

    # Check if it is a XGBClassificationModel
    elif str_len >= 5 and prefix_or_id[0:5] == "XGBC_":
        return "XGBC_"

    # Check if it is a ConsensusModel
    elif str_len >= 4 and prefix_or_id[0:4] == "CON_":
        return "CON_"
    
    # Invalid ID
    else:
        raise ValueError(f"A model prefix could not be extracted from the provided id: {prefix_or_id}")
