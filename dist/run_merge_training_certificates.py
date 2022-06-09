from typing import List, Dict
from os import walk
from os.path import exists, isfile
from json import load, dumps
from inquirer import List as InquirerList, prompt
from modules.utils import Utils
from modules.keras_models import KERAS_PATH
from modules.classification import IClassificationTrainingCertificate

# Configuration Input
print("MERGE TRAINING CERTIFICATES")
answers: List[Dict[str, str]] = prompt([InquirerList("prefix", message="Select the certificates type", choices=["R_", "C_"])])


# Model IDs Extractor
def _get_model_ids(prefix: str) -> List[str]:
    # Make sure the directory exists
    if not exists(KERAS_PATH["models"]):
        raise RuntimeError("The keras_assets/models directory does not exist.")

    # Extract all the directories within
    dirs: List[str] = [dir[0].split("/")[2] if len(dir[0].split("/")) == 3 else "" for dir in walk(KERAS_PATH["models"])]

    # Return the filtered list
    def _is_certificate(id: str) -> bool:
        return len(id) > 0 and id[0:2] == prefix and "UNIT_TEST" not in id and isfile(f"{KERAS_PATH['models']}/{id}/{id}.json")
    return list(filter(lambda d: _is_certificate(d), dirs))


# Build the list of model id's
ids: List[str] = _get_model_ids(answers["prefix"])
if len(ids) == 0:
    raise RuntimeError("No model ids could be extracted.")

# Extract the certificates json and place them in a list
certificates: List[IClassificationTrainingCertificate] = [load(open(f"{KERAS_PATH['models']}/{id}/{id}.json")) for id in ids]

# Finally, dump the merged file in the configs directory
path: str = f"{KERAS_PATH['batched_training_certificates']}/{answers['prefix']}MERGE_{Utils.get_time()}.json"
with open(path, "w") as outfile:
    outfile.write(dumps(certificates))
print("\nMERGE TRAINING CERTIFICATES COMPLETED")