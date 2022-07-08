from typing import List, Dict, Union
from os import walk, makedirs
from os.path import exists, isfile
from json import load, dumps
from inquirer import List as InquirerList, prompt
from modules.types import IRegressionTrainingCertificate, IClassificationTrainingCertificate
from modules.epoch.Epoch import Epoch
from modules.utils.Utils import Utils
from modules.keras_models.KerasPath import KERAS_PATH


# Initialize the Epoch
Epoch.init()


# Configuration Input
print("MERGE TRAINING CERTIFICATES")
answers: Dict[str, str] = prompt([
    InquirerList(
        "model_type", 
        message="Select the type of model", 
        choices=["KerasRegression", "KerasClassification"])
])
model_type: str = answers["model_type"]
prefix: str = "R_" if model_type == "KerasRegression" else "C_"




# Model IDs Extractor
def _get_model_ids(prefix: str) -> List[str]:
    # Make sure the directory exists
    if not exists(KERAS_PATH["models"]):
        raise RuntimeError("The keras_assets/models directory does not exist.")

    # Extract all the directories within
    dirs: List[str] = [dir[0].split("/")[2] if len(dir[0].split("/")) == 3 else "" for dir in walk(KERAS_PATH["models"])]

    # Return the filtered list
    def _is_certificate(id: str) -> bool:
        return len(id) > 0 and id[0:2] == prefix and "UNIT_TEST" not in id and isfile(f"{KERAS_PATH['models']}/{id}/certificate.json")
    return list(filter(lambda d: _is_certificate(d), dirs))


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
path: str = f"{KERAS_PATH['batched_training_certificates']}/{model_type}_MERGE_{Utils.get_time()}.json"
with open(path, "w") as outfile:
    outfile.write(dumps(certificates))


print("\nMERGE TRAINING CERTIFICATES COMPLETED")