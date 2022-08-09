from typing import List, Dict, Union, Any
from os import walk, makedirs
from os.path import exists, isfile
from json import load, dumps
from copy import deepcopy
from shutil import rmtree
from inquirer import List as InquirerList, prompt
from modules.types import IRegressionTrainingCertificate, IClassificationTrainingCertificate, \
    ITrainableModelType, IModelIDPrefix, IRegressionTrainingBatch, IClassificationTrainingBatch,\
        IRegressionTrainingConfig, IClassificationTrainingConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.model.ModelType import TRAINABLE_MODEL_TYPES


# Initialize the Epoch
Epoch.init()


# Welcome
print("MERGE TRAINING CERTIFICATES")



# Model Type Input
# Populate all required values based on the type of model.
model_type_answer: Dict[str, str] = prompt([InquirerList("model_type", message="Select the type of model", choices=TRAINABLE_MODEL_TYPES)])
model_type: ITrainableModelType = model_type_answer["model_type"]




# Original Training Configuration File
# Load the original configuration file in order to extract the name of the batch
# and be able to clear the models that have already been trained.
print("\n1/10) Extracting the original configuration file...")
original_config: Union[IRegressionTrainingBatch, IClassificationTrainingBatch, Any]
if model_type == "keras_regression":
    original_config = Epoch.FILE.get_keras_regression_training_config()
elif model_type == "keras_classification":
    original_config = Epoch.FILE.get_keras_classification_training_config()
elif model_type == "xgb_regression":
    original_config = Epoch.FILE.get_xgb_regression_training_config()
elif model_type == "xgb_classification":
    original_config = Epoch.FILE.get_xgb_classification_training_config()
else:
    raise ValueError(f"Could not retrieve the configuration for: {model_type}")




# Model IDs Extractor
# Extracts all the ids of the models that completed training. The ones that did not complete 
# the training or the evaluation are deleted.
def _get_model_ids(prefix: str) -> List[str]:
    # Make sure the directory exists
    if not exists(KERAS_PATH["models"]):
        raise RuntimeError("The keras_assets/models directory does not exist.")

    # Extract all the directories within
    dirs: List[str] = [dir[0].split("/")[2] if len(dir[0].split("/")) == 3 else "" for dir in walk(KERAS_PATH["models"])]

    # Iterate over each directory within the models directory
    model_ids: List[str] = []
    for dir in dirs:
        # Make sure the directory matches with the type of model that is being merged
        if len(dir) >= 3 and prefix in dir and "UNIT_TEST" not in dir:
            # Check if the model completed the evaluation
            if isfile(f"{KERAS_PATH['models']}/{dir}/certificate.json"):
                model_ids.append(dir)
            
            # Otherwise, remove the directory and the trained model
            else:
                rmtree(f"{KERAS_PATH['models']}/{dir}")

    # Finally, return the list of model ids
    return model_ids




# Build the list of model id's
ids: List[str] = _get_model_ids(prefix)
if len(ids) == 0:
    raise RuntimeError("No model ids could be extracted.")



# Extract the certificates json and place them in a list
certificates: Union[List[IRegressionTrainingCertificate], List[IClassificationTrainingCertificate]] = \
    [load(open(f"{KERAS_PATH['models']}/{id}/certificate.json")) for id in ids]




# Dump the merged file in the output directory
if not exists(KERAS_PATH["batched_training_certificates"]):
    makedirs(KERAS_PATH["batched_training_certificates"])
path: str = f"{KERAS_PATH['batched_training_certificates']}/{original_config['name']}_MERGE_{Utils.get_time()}.json"
with open(path, "w") as outfile:
    outfile.write(dumps(certificates))



# Update the training configuration file leaving only the models that did not complete
new_config: Union[IRegressionTrainingBatch, IClassificationTrainingBatch] = deepcopy(original_config)
remaining_models: Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]] = []
for model_config in new_config["models"]:
    if model_config["id"] not in ids:
        remaining_models.append(model_config)
new_config["models"] = remaining_models
with open(f"config/{config_file_name}", "w") as config_file:
    config_file.write(dumps(new_config, indent=4))



# Execution Completed
print("\n\nMERGE TRAINING CERTIFICATES COMPLETED")