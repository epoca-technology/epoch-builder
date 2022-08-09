from typing import List, Dict
from inquirer import List as InquirerList, Text, prompt
from modules._types import IRegressionSelectionFile
from modules.epoch.Epoch import Epoch
from modules.classification_training_data.ClassificationTrainingData import ClassificationTrainingData


# EPOCH INIT
Epoch.init()



## WELCOME ##
print("CLASSIFICATION TRAINING DATA\n")




# Regression Selection
print(" ")
regression_selection_ids: List[str] = Epoch.FILE.list_regression_selection_ids()
if len(regression_selection_ids) == 0:
    raise RuntimeError("The Regression Selection assets directory is empty.")
regression_selection_answer: Dict[str, str] = prompt([InquirerList("id", message="Select the Regression Selection", choices=regression_selection_ids)])
regression_selection: IRegressionSelectionFile = Epoch.FILE.get_regression_selection(regression_selection_answer["id"])


# Description
description_answer: Dict[str, str] = prompt([Text("content", f"Enter the description")])
description: str = description_answer["content"]


# Steps
print(" ")
steps_answer: Dict[str, str] = prompt([Text("value", f"Enter the steps")])
if not steps_answer["value"].isdigit():
    raise ValueError("The steps must be a valid integer.")
steps: int = int(steps_answer["value"])


# RSI
print(" ")
include_rsi_answer: Dict[str, str] = prompt([InquirerList("value", message="Include RSI", choices=["No", "Yes"])])
include_rsi: bool = include_rsi_answer["value"] == "Yes"


# AROON
include_aroon_answer: Dict[str, str] = prompt([InquirerList("value", message="Include AROON", choices=["No", "Yes"])])
include_aroon: bool = include_aroon_answer["value"] == "Yes"



# TRAINING DATA INSTANCE
# The Instance of the Training Data that will be executed
training_data: ClassificationTrainingData = ClassificationTrainingData({
    "regression_selection_id": regression_selection["id"],
    "description": description,
    "steps": steps,
    "price_change_requirement": regression_selection["price_change_mean"],
    "regressions": [selected_model["model"] for selected_model in regression_selection["selection"]],
    "include_rsi": include_rsi,
    "include_aroon": include_aroon
})



# TRAINING DATA EXECUTION
# Runs the Training Data for all models simultaneously. 
# Results as saved when the execution has completed. If the execution is interrupted before
# completion, results will not be saved.
print("\nCLASSIFICATION TRAINING DATA RUNNING\n")
print(f"{training_data.id}:\n")
training_data.run()
print("\n\nnCLASSIFICATION TRAINING DATA COMPLETED")
