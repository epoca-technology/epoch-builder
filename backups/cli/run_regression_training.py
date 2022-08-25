from typing import List, Dict, Union
from inquirer import List as InquirerList, prompt
from modules._types import IKerasRegressionTrainingBatch, IKerasRegressionTrainingCertificate, ITrainableModelType,\
    IKerasRegressionTrainingConfig, IXGBRegressionTrainingBatch, IXGBRegressionTrainingCertificate, IXGBRegressionTrainingConfig,\
        IHyperparamsCategory
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.model.ModelType import TRAINABLE_REGRESSION_MODEL_TYPES
from modules.keras_regression.KerasRegressionTraining import KerasRegressionTraining
from modules.xgb_regression.XGBRegressionTraining import XGBRegressionTraining


## Types Helpers ##

# Training Batch
ITrainingBatch = Union[IKerasRegressionTrainingBatch, IXGBRegressionTrainingBatch]


# Training Configuration
ITrainingConfig = Union[IKerasRegressionTrainingConfig, IXGBRegressionTrainingConfig]


# Trainer Instance
ITrainer = Union[KerasRegressionTraining, XGBRegressionTraining]


# Training Certificate
ITrainingCertificate = Union[IKerasRegressionTrainingCertificate, IXGBRegressionTrainingCertificate]




# EPOCH INIT
Epoch.init()




# MODEL TYPE
# The trainable model type required to retrieve the configuration, run the training
# and store the results.
print("REGRESSION TRAINING\n")
model_type_answer: Dict[str, str] = prompt([InquirerList("value", message="Select the type of model", choices=TRAINABLE_REGRESSION_MODEL_TYPES)])
model_type: ITrainableModelType = model_type_answer["value"]



# CATEGORY
# The category in which the desired batch configuration file is stored.
Utils.clear_terminal()
print("KERAS REGRESSION TRAINING\n" if model_type == "keras_regression" else "XGBOOST REGRESSION TRAINING\n")
categories: List[IHyperparamsCategory] = Epoch.FILE.list_training_config_categories(model_type)
category_answer: Dict[str, str] = prompt([InquirerList("value", message="Select the category", choices=categories)])
category: IHyperparamsCategory = category_answer["value"]



# TRAINING CONFIGURATION FILE NAME
# The name of the batch configuration file that will be put through the training process.
config_file_name: str

# If it is the unit test, populate the file name right away
if category == "UNIT_TEST":
    config_file_name = "UNIT_TEST.json"

# Otherwise, show the config files within the category
else:
    Utils.clear_terminal()
    print("KERAS REGRESSION TRAINING\n" if model_type == "keras_regression" else "XGBOOST REGRESSION TRAINING\n")
    configs: List[str] = Epoch.FILE.list_training_config_names(model_type, category)
    config_answer: Dict[str, str] = prompt([InquirerList("value", message="Select the configuration", choices=configs)])
    config_file_name = config_answer["value"]




# TRAINING CONFIGURATION
# If the selected configuration file is within the root config directory, it will just load it and 
# use it. Otherwise, it will grab a copy from the epoch directory and place it in the root config.

# Init the final path for the config file
config_file_path: str = f"{Configuration.DIR_PATH}/{config_file_name}"

# If the file is not in the config directory, grab a copy
if not Utils.file_exists(config_file_path):
    Utils.copy_file_or_dir(
        source=Epoch.FILE.get_training_config_path(model_type, category, config_file_name),
        destination=config_file_path
    )

# Read the batch configuration file from the root directory
config: ITrainingBatch = Utils.read(config_file_path)




# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the regression lookback.
Candlestick.init(Epoch.REGRESSION_LOOKBACK, Epoch.START, Epoch.END)




# CERTIFICATES
# This file will be used to create the certificates batch once all models finish training.
certificates: Union[List[IKerasRegressionTrainingCertificate], List[IXGBRegressionTrainingCertificate]] = []




# TRAINER INSTANCE
# Returns the instance of a training class based on the selected model_type
def get_trainer(model_config: ITrainingConfig) -> ITrainer:
    if model_type == "keras_regression":
        return KerasRegressionTraining(model_config)
    elif model_type == "xgb_regression":
        return XGBRegressionTraining(model_config)
    else:
        raise ValueError(f"Could not find the training class for model_type: {model_type}")




# REGRESSION TRAINING EXECUTION
# Builds, trains, saves and evaluates models. Once models finish training, their file as well as the
# certificate is saved in the models directory. 
# When the process completes, it saves all the certificates into a single batch and then moves
# the models to the models_bank directory.
# If the training is interrupted for any reason, make use of the Merge Training Certificates
# script in order to merge the certificates of the models that finished and then move them to
# the models_bank directory. Notice that when this script is executed, it also modifies the 
# corresponding Training Configuration File, leaving only the models that did not finish.
for index, model_config in enumerate(config["models"]):
    # Initialize the instance of the trainer
    trainer: ITrainer = get_trainer(model_config)

    # Log the progress
    if index == 0:
        Utils.clear_terminal()
        print("KERAS REGRESSION TRAINING\n" if model_type == "keras_regression" else "XGBOOST REGRESSION TRAINING\n")
        print(f"{config['name']}\n")
    print(f"\n{index + 1}/{len(config['models'])}) {Utils.prettify_model_id(model_config['id'])}")

    # Train the model
    cert: ITrainingCertificate = trainer.train()

    # Add the certificate to the list
    certificates.append(cert)



# CERTIFICATE BATCH SAVING
Epoch.FILE.save_training_certificate_batch(model_type, config["name"], certificates)



# MOVING MODELS TO BANK
Epoch.FILE.move_trained_models_to_bank(model_type, certificates)



# REMOVING TEMP CONFIG FILE
Utils.remove_file(f"{Configuration.DIR_PATH}/{config_file_name}")



print("\n\nKERAS REGRESSION TRAINING COMPLETED\n" if model_type == "keras_regression" else "\n\nXGBOOST REGRESSION TRAINING COMPLETED\n")