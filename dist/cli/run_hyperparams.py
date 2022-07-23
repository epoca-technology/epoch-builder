from typing import Dict, Union
from inquirer import Text, List as InquirerList, prompt
from modules.types import ITrainableModelType
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.model.ModelType import TRAINABLE_MODEL_TYPES
from modules.hyperparams.KerasHyperparams import KerasHyperparams


# Initialize the Epoch
Epoch.init()


# Welcome
print("HYPERPARAMS\n\n")




# Main Inputs - The type of model to generate hyperparams for
print(" ")
model_type_answers: Dict[str, str] = prompt([InquirerList("model_type", message="Select the type of model", choices=TRAINABLE_MODEL_TYPES)])
model_type: ITrainableModelType = model_type_answers["model_type"]



# Batch Size - The maximum amount of models that will be included per batch.
print(" ")
default_batch_size: int
if model_type == "keras_regression":
    default_batch_size = KerasHyperparams.REGRESSION_BATCH_SIZE
elif model_type == "keras_classification":
    default_batch_size = KerasHyperparams.CLASSIFICATION_BATCH_SIZE
else:
    raise ValueError(f"The default batch size could not be extracted for: {model_type}")
batch_size_answer: Dict[str, str] = prompt([Text("size", f"Enter the Batch Size (Defaults to {default_batch_size})")])
batch_size: int = int(batch_size_answer["size"]) if batch_size_answer["size"].isdigit() else default_batch_size

    


# Training Data ID - Only applies to Classifications
training_data_id: Union[str, None] = None
if model_type == "keras_classification" or model_type == "xgb_classification":
    print(" ")
    training_data: Dict[str, str] = prompt([Text("id", "Enter the Training Data ID")])
    training_data_id = training_data["id"]
    if not Utils.is_uuid4(training_data_id):
        raise ValueError("The provided training data id is invalid.")





# Hyperparams Generation
print("\n\HYPERPARAMS RUNNING\n")


# Keras Hyperparams
if model_type == "keras_regression" or model_type == "keras_classification":
    # Initialize the Hyperparams Instance
    hyperparams: KerasHyperparams = KerasHyperparams(model_type, batch_size, training_data_id)

    # Generate the configurations
    hyperparams.generate()


# XGBoost Hyperparams
elif model_type == "xgb_regression" or model_type == "xgb_classification":
    raise NotImplementedError("XGBoost Hyperparams has not yet been implemented.")


# Unknown model type
else:
    raise ValueError("The provided type of model is invalid.")




# Execution Completed
print("\n\nHYPERPARAMS COMPLETED")