from typing import Union
from argparse import ArgumentParser
from modules._types import ITrainableModelType
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.hyperparams.KerasHyperparams import KerasHyperparams



# HYPERPARAMS
# Args:
#   --model_type "keras_regression"
#   --training_data_file_name? "ea998c4d-9142-435f-8cc9-c5804ed5c1e8.json" -> Mandatory in keras_classification|xgb_classification
#   --batch_size? "30"
endpoint_name: str = "HYPERPARAMS"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)




# Initialize the Epoch
Epoch.init()




# Extract the args
parser = ArgumentParser()
parser.add_argument("--model_type", dest="model_type")
parser.add_argument("--batch_size", dest="batch_size", nargs='?')
args = parser.parse_args()



# The type of model to generate hyperparams for
model_type: ITrainableModelType = args.model_type



# Batch Size - The maximum amount of models that will be included per batch.
default_batch_size: int
if model_type == "keras_regression":
    default_batch_size = KerasHyperparams.REGRESSION_BATCH_SIZE
elif model_type == "keras_classification":
    default_batch_size = KerasHyperparams.CLASSIFICATION_BATCH_SIZE
else:
    raise ValueError(f"The default batch size could not be extracted for: {model_type}")
batch_size: int = int(args.batch_size) if args.batch_size.isdigit() else default_batch_size





# Training Data ID - Only applies to Classifications
training_data_id: Union[str, None] = None
if model_type == "keras_classification" or model_type == "xgb_classification":
    # Init the training data id
    training_data_id = args.training_data_file_name.replace(".json", "")

    # Make sure the file exists. If it doesn't it will throw an error
    Epoch.FILE.get_classification_training_data(training_data_id)







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




# End of Script
Utils.endpoint_footer(endpoint_name)