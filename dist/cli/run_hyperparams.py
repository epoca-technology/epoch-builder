from typing import List, Dict, Union
from inquirer import Text, List as InquirerList, prompt
from modules.utils import Utils
from modules.hyperparams import KerasHyperparams

# Configuration Input
print("HYPERPARAMS\n")

# Output Name
output: Dict[str, str] = prompt([Text("name", "Enter the name of the output directory")])
output_name: str = output["name"]
if len(output_name) < 4:
    raise ValueError("The name of the output directory must be at least 4 characters long.")


# Main Inputs
print(" ")
answers: Dict[str, str] = prompt([
    # The type of model to generate hyperparams for
    InquirerList(
        "model_type", 
        message="Select the type of model", 
        choices=["KerasRegression", "KerasClassification"]
    ),

    # The maximum amount of models that will be included per batch.
    Text("batch_size", f"Enter the Batch Size (Defaults to {KerasHyperparams.DEFAULT_BATCH_SIZE})")
])
model_type: str = answers["model_type"]
batch_size: int = int(answers["batch_size"]) if answers["batch_size"].isdigit() else KerasHyperparams.DEFAULT_BATCH_SIZE



# Start and End Date - Only applies to Regressions
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
if model_type == "KerasRegression" or model_type == "KerasClassification":
    # Initialize the Hyperparams Instance
    hyperparams: KerasHyperparams = KerasHyperparams(
        output_name, 
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