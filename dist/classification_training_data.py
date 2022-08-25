from argparse import ArgumentParser
from modules._types import IRegressionSelectionFile
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.classification_training_data.ClassificationTrainingData import ClassificationTrainingData



# CLASSIFICATION TRAINING DATA
# Args:
#   --regression_selection_file_name "bfe1ff30-0997-4224-aede-8ed9c0abbb8d.json"
#   --description "Some important information regarding the training data."
#   --steps "5"
#   --include_rsi "Yes"|"No"
#   --include_aroon "Yes"|"No"
endpoint_name: str = "CLASSIFICATION TRAINING DATA"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# EPOCH INIT
Epoch.init()



# Extract the args
parser = ArgumentParser()
parser.add_argument("--regression_selection_file_name", dest="regression_selection_file_name")
parser.add_argument("--description", dest="description")
parser.add_argument("--steps", dest="steps")
parser.add_argument("--include_rsi", dest="include_rsi")
parser.add_argument("--include_aroon", dest="include_aroon")
args = parser.parse_args()




# Regression Selection
regression_selection: IRegressionSelectionFile = Epoch.FILE.get_regression_selection(args.regression_selection_file_name.replace(".json", ""))



# Steps
print(" ")
if not args.steps.isdigit():
    raise ValueError("The steps must be a valid integer.")
steps: int = int(args.steps)



# Technical Analysis Features
include_rsi: bool = args.include_rsi == "Yes"
include_aroon: bool = args.include_aroon == "Yes"



# TRAINING DATA INSTANCE
# The Instance of the Training Data that will be executed
training_data: ClassificationTrainingData = ClassificationTrainingData({
    "regression_selection_id": regression_selection["id"],
    "description": args.description,
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
print(f"Running {training_data.id}:\n")
training_data.run()




# End of Script
Utils.endpoint_footer(endpoint_name)