from typing import Dict, Union
from inquirer import Text, List as InquirerList, prompt
from modules.types import ITrainableModelType
from modules.epoch.Epoch import Epoch
from modules.utils.Utils import Utils
from modules.hyperparams.KerasHyperparams import KerasHyperparams


# Initialize the Epoch
Epoch.init()


# Configuration Input
print("HYPERPARAMS\n")

# Epoch Name - @DEPRECATE
epoch: Dict[str, str] = prompt([Text("name", "Enter the name of the epoch")])
epoch_name: str = epoch["name"]
if len(epoch_name) < 4 or epoch_name[0] != "_":
    raise ValueError("The name of the epoch must be at least 4 characters long and must be prefixed with '_EPOCHNAME'.")


# Main Inputs - The type of model to generate hyperparams for
print(" ")
model_type_answers: Dict[str, str] = prompt([
    InquirerList(
        "model_type", 
        message="Select the type of model", 
        choices=["keras_regression", "keras_classification"]
    )
])
model_type: ITrainableModelType = model_type_answers["model_type"]



# Batch Size - The maximum amount of models that will be included per batch.
print(" ")
default_batch_size: int
if model_type == "keras_regression":
    default_batch_size = KerasHyperparams.REGRESSION_BATCH_SIZE
elif model_type == "keras_classification":
    default_batch_size = KerasHyperparams.CLASSIFICATION_BATCH_SIZE
batch_size_answer: Dict[str, str] = prompt([
    Text("size", f"Enter the Batch Size (Defaults to {default_batch_size})")
])
batch_size: int = int(batch_size_answer["size"]) if batch_size_answer["size"].isdigit() else default_batch_size




# Start and End Date - Only applies to Regressions - DEPRECATE
start: Union[str, None] = None
end: Union[str, None] = None
if "Regression" in model_type:
    print(" ")
    date_range: Dict[str, str] = prompt([
        Text("start", "Enter the Start Date 'DD/MM/YYYY'"),
        Text("end", "Enter the End Date (Optional)")
    ])
    start = date_range["start"]
    if len(start) != 10:
        raise ValueError("The start date must be a string in the following format: DD/MM/YYYY")
    if len(date_range["end"]) != 0 and len(date_range["end"]) != 10:
        raise ValueError("The end date must be a string in the following format: DD/MM/YYYY")
    elif len(date_range["end"]) == 10:
        end = date_range["end"]

    




# Training Data ID - Only applies to Classifications
training_data_id: Union[str, None] = None
if "Classification" in model_type:
    print(" ")
    training_data: Dict[str, str] = prompt([Text("id", "Enter the Training Data ID")])
    training_data_id = training_data["id"]
    if not Utils.is_uuid4(training_data_id):
        raise ValueError("The provided training data id is invalid.")





# Hyperparams Generation
if model_type == "keras_regression" or model_type == "keras_classification":
    # Initialize the Hyperparams Instance
    hyperparams: KerasHyperparams = KerasHyperparams(
        epoch_name, 
        model_type, 
        batch_size, 
        start,
        end,
        training_data_id
    )

    # Generate the configurations
    hyperparams.generate()

    # Print the summary of the generated files
    print("\n\nOutput:")
    print(f"{hyperparams.output_path}/receipt.txt")
    for net in hyperparams.networks:
        print(f"{hyperparams.output_path}/{net}/...")
else:
    raise ValueError("The provided type of model is invalid.")




# Execution Completed
print("\n\nHYPERPARAMS COMPLETED")