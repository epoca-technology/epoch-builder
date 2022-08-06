from typing import List, Dict, Union
from copy import deepcopy
from inquirer import List as InquirerList, prompt
from modules._types import IKerasRegressionTrainingCertificate, IKerasClassificationTrainingCertificate, \
    ITrainableModelType, IKerasRegressionTrainingBatch, IKerasClassificationTrainingBatch,\
        IKerasRegressionTrainingConfig, IKerasClassificationTrainingConfig, IXGBRegressionTrainingCertificate,\
            IXGBClassificationTrainingCertificate, IXGBRegressionTrainingBatch, IXGBClassificationTrainingBatch,\
                IXGBRegressionTrainingConfig, IXGBClassificationTrainingConfig
from modules.epoch.Epoch import Epoch
from modules.model.ModelType import TRAINABLE_MODEL_TYPES



## Trainable Model Type Helpers ##


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






# Initialize the Epoch
Epoch.init()


# Welcome
print("MERGE TRAINING CERTIFICATES\n")



# Model Type Input
# Populate all required values based on the type of model.
model_type_answer: Dict[str, str] = prompt([InquirerList("model_type", message="Select the type of model", choices=TRAINABLE_MODEL_TYPES)])
model_type: ITrainableModelType = model_type_answer["model_type"]




# Original Training Configuration File
# Load the original configuration file in order to extract the name of the batch
# and be able to clear the models that have already been trained.
print("\n1/6) Extracting the original configuration file...")
original_config: ITrainingBatch
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



# Retrieve the active model ids
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
model_ids: List[str] = _get_model_ids()



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
if model_type == "keras_regression":
    Epoch.FILE.update_keras_regression_training_config(new_config)
elif model_type == "keras_classification":
    Epoch.FILE.update_keras_classification_training_config(new_config)
elif model_type == "xgb_regression":
    Epoch.FILE.update_xgb_regression_training_config(new_config)
elif model_type == "xgb_classification":
    Epoch.FILE.update_xgb_classification_training_config(new_config)
else:
    raise ValueError(f"Could not update the configuration for: {model_type}")


# Finally, move the models to the bank
print("6/6) Moving the trained models to the bank...")
Epoch.FILE.move_trained_models_to_bank(model_type, certificates)



# Execution Completed
print("\n\nMERGE TRAINING CERTIFICATES COMPLETED")