from typing import List, Tuple, Dict, Union, Any
from pandas import DataFrame
from inquirer import List as InquirerList, prompt
from modules.types import IClassificationTrainingBatch, IClassificationTrainingCertificate, ITrainingDataFile,\
    ITrainableModelType, IClassificationTrainingConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.model.ModelType import TRAINABLE_CLASSIFICATION_MODEL_TYPES
from modules.classification.ClassificationTraining import ClassificationTraining
    

# EPOCH INIT
Epoch.init()



## WELCOME ##
print("CLASSIFICATION TRAINING\n")
model_type_answer: Dict[str, str] = prompt([InquirerList("type", message="Select the type of model", choices=TRAINABLE_CLASSIFICATION_MODEL_TYPES)])
model_type: ITrainableModelType = model_type_answer["type"]



# CLASSIFICATION TRAINING CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config: Union[IClassificationTrainingBatch, Any]
if model_type == "keras_classification":
    config = Epoch.FILE.get_keras_classification_training_config()
elif model_type == "xgb_classification":
    config = Epoch.FILE.get_xgb_classification_training_config()
else:
    raise ValueError(f"The provided type of model could not be processed.")



# TRAINING DATA FILE
# Opens and loads the training_data file that will be used to train the models.
# IMPORTANT: All classification training configs must have the "name" and 
# "training_data_id" properties.
training_data: ITrainingDataFile = Epoch.FILE.get_classification_training_data(config["training_data_id"])



# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the highest lookback among the models config.
Candlestick.init(300, start=training_data["start"], end=training_data["end"])



# DATASETS
# Initialize the train and test datasets for the entire batch of models that will be trained.
datasets: Tuple[DataFrame, DataFrame, DataFrame, DataFrame] = ClassificationTraining.make_datasets(
    training_data=training_data["training_data"]
)



# CERTIFICATES
# This file will be used to create the certificates batch once all models finish training.
certificates: Union[List[IClassificationTrainingCertificate], List[Any]] = []



# TRAINER INSTANCE
# Returns the instance of a training class based on the selected model_type
def get_trainer(model_config: Union[IClassificationTrainingConfig, Any]) -> Union[ClassificationTraining, Any]:
    if model_type == "keras_classification":
        return ClassificationTraining(training_data_file=training_data, config=model_config, datasets=datasets)
    elif model_type == "xgb_classification":
        raise ValueError("XGBClassification Models cannot be trained as they have not been yet implemented.")
    else:
        raise ValueError(f"Could not find the training class for model_type: {model_type}")



# CLASSIFICATION TRAINING EXECUTION
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
    trainer: Union[ClassificationTraining, Any] = get_trainer(model_config)

    # Log the progress
    if index == 0:
        print("\nCLASSIFICATION TRAINING RUNNING\n")
        print(f"{config['name']}\n")
    print(f"\n{index + 1}/{len(config['models'])}) {Utils.prettify_model_id(model_config['id'])}")

    # Train the model
    cert: Union[IClassificationTrainingCertificate, Any] = trainer.train()

    # Add the certificate to the list
    certificates.append(cert)



# CERTIFICATE BATCH SAVING
Epoch.FILE.save_training_certificate_batch(model_type, config["name"], certificates)



# MOVING MODELS TO BANK
Epoch.FILE.move_trained_models_to_bank(model_type, certificates)



print("\n\nCLASSIFICATION TRAINING COMPLETED")