from typing import List, Union
from argparse import ArgumentParser
from copy import deepcopy
from modules._types import IKerasRegressionTrainingCertificate, IKerasClassificationTrainingCertificate, \
    ITrainableModelType, IKerasRegressionTrainingBatch, IKerasClassificationTrainingBatch,\
        IKerasRegressionTrainingConfig, IKerasClassificationTrainingConfig, IXGBRegressionTrainingCertificate,\
            IXGBClassificationTrainingCertificate, IXGBRegressionTrainingBatch, IXGBClassificationTrainingBatch,\
                IXGBRegressionTrainingConfig, IXGBClassificationTrainingConfig, IHyperparamsCategory
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch




# MERGE TRAINING CERTIFICATES
# Args:
#   --model_type "keras_regression"
endpoint_name: str = "MERGE TRAINING CERTIFICATES"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)




## Type Helpers ##


# Training Config List
ITrainingConfigList = Union[
    List[IKerasRegressionTrainingConfig], 
    List[IKerasClassificationTrainingConfig], 
    List[IXGBRegressionTrainingConfig],
    List[IXGBClassificationTrainingConfig]
]

# Training Batch
ITrainingBatch = Union[
    IKerasRegressionTrainingBatch, IKerasClassificationTrainingBatch, 
    IXGBRegressionTrainingBatch, IXGBClassificationTrainingBatch
]

# Training Certificate Lists
ITrainingCertificateList = Union[
    List[IKerasRegressionTrainingCertificate], List[IKerasClassificationTrainingCertificate],
    List[IXGBRegressionTrainingCertificate], List[IXGBClassificationTrainingCertificate]
]



# Function Helpers


def _get_category(file_name: str) -> Union[IHyperparamsCategory, None]:
    """Based on a given file name, it will attempt to extract the 
    category name. If nothing is found, it returns None.

    Args:
        file_name: str
            The name of the file to be checked
    """
    if "DNN" in file_name:
        return "DNN"
    elif "CNN" in file_name:
        return "CNN"
    elif "LSTM" in file_name:
        return "LSTM"
    elif "CLSTM" in file_name:
        return "CLSTM"
    else:
        return None






# Initialize the Epoch
Epoch.init()




# Extract the args
parser = ArgumentParser()
parser.add_argument("--model_type", dest="model_type")
args = parser.parse_args()





# MERGEABLE CONFIGURATIONS
# Reads the config directory and extracts all the possible configuration files that
# can be merged.

# Retrieve the config contents
_, files = Utils.get_directory_content(Configuration.DIR_PATH, only_file_ext=".json")

# Filter all the files that are not training configurations
config_names: List[str] = list(filter(lambda x: isinstance(_get_category(x), str), files))
if len(config_names) == 0:
    raise RuntimeError("No mergeable configurations were found in the root config directory.")
elif len(config_names) > 1:
    print(config_names)
    raise RuntimeError("Cannot merge certificates when there are more than 1 mergeable configurations.")





# TRAINABLE MODEL TYPE
# The type of model which certificates will be merged.
model_type: ITrainableModelType = args.model_type
if not Utils.file_exists(Epoch.FILE.get_training_config_path(model_type, _get_category(config_names[0]), config_names[0])):
    raise RuntimeError(f"The file {model_type}/{config_names[0]} does not exist. Make sure the correct option has been selected.")






# ORIGINAL TRAINING CONFIGURATION FILE
# Load the original configuration file in order to extract the name of the batch
# and be able to clear the models that have already been trained.
print("\n1/6) Extracting the original configuration file...")
original_config: ITrainingBatch = Utils.read(f"{Configuration.DIR_PATH}/{config_names[0]}")
if not isinstance(original_config.get("name"), str) or \
    not isinstance(original_config.get("models"), list) or \
        len(original_config["models"]) == 0:
    print(original_config)
    raise RuntimeError("The extracted batch configuration is invalid or has no models in it.")





# ACTIVE MODEL IDS
# In order to perform the merge of the current training process, the list of models that 
# completed training, discovery and evaluation are merged, saved and moved to the bank.
print("2/6) Extracting the trained models...")
def _get_model_ids() -> List[str]:
    # Init values
    ids: List[str] = []

    # Extract the ids except for the unit test
    raw_ids: List[str] = Epoch.FILE.get_active_model_ids(model_type, exclude_unit_test=True)

    # Iterate over each raw id
    for id in raw_ids:
        # Check if the model has a certificate
        if Epoch.FILE.active_model_has_certificate(id):
            ids.append(id)

        # Otherwise, it means the model did not finish the training process and must be deleted
        else:
            Epoch.FILE.remove_active_model(id)

    # Make sure at least 1 model was extracted
    if len(ids) == 0:
        raise RuntimeError(f"No models could be extracted for {model_type}.")

    # Finally, return the ids
    return ids

# Retrieve the list of ids and ake sure that each of the active models exists in the configuration file
model_ids: List[str] = _get_model_ids()
for id in model_ids:
    found: bool = False
    for model in original_config["models"]:
        if id == model["id"]:
            found = True
    if not found:
        raise RuntimeError(f"The model {id} was not found in the configuration file.")



# Extract the certificates and place them in a list
print("3/6) Extracting certificates...")
certificates:ITrainingCertificateList = [Epoch.FILE.get_active_model_certificate(id) for id in model_ids]



# Save the merge batch
print("4/6) Saving merged batch...")
Epoch.FILE.save_training_certificate_batch(model_type, original_config["name"] + "_MERGE", certificates)



# Update the training configuration file leaving only the models that did not complete
print("5/6) Updating the training configuration...")
new_config: ITrainingBatch = deepcopy(original_config)
remaining_models: ITrainingConfigList = []
for model_config in new_config["models"]:
    if model_config["id"] not in model_ids:
        remaining_models.append(model_config)
new_config["models"] = remaining_models

# If there are remaining models, update the configuration file
config_path: str = f"{Configuration.DIR_PATH}/{config_names[0]}"
if len(remaining_models) > 0:
    Utils.write(config_path, new_config, indent=4)

# Otherwise, delete the configuration file
else:
    Utils.remove_file(config_path)




# Finally, move the models to the bank
print("6/6) Moving the trained models to the bank...")
Epoch.FILE.move_trained_models_to_bank(model_type, certificates)





# End of Script
Utils.endpoint_footer(endpoint_name)