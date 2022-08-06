from typing import List, Dict, Union
from inquirer import List as InquirerList, prompt
from modules._types import IKerasRegressionTrainingBatch, IKerasRegressionTrainingCertificate, ITrainableModelType,\
    IKerasRegressionTrainingConfig, IXGBRegressionTrainingBatch, IXGBRegressionTrainingCertificate, IXGBRegressionTrainingConfig
from modules.epoch.Epoch import Epoch
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.model.ModelType import TRAINABLE_REGRESSION_MODEL_TYPES
from modules.keras_regression.KerasRegressionTraining import KerasRegressionTraining
from modules.xgb_regression.XGBRegressionTraining import XGBRegressionTraining



# EPOCH INIT
Epoch.init()



# WELCOME
print("REGRESSION TRAINING\n")
model_type_answer: Dict[str, str] = prompt([InquirerList("type", message="Select the type of model", choices=TRAINABLE_REGRESSION_MODEL_TYPES)])
model_type: ITrainableModelType = model_type_answer["type"]



# TRAINING CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config: Union[IKerasRegressionTrainingBatch, IXGBRegressionTrainingBatch]
if model_type == "keras_regression":
    config = Epoch.FILE.get_keras_regression_training_config()
elif model_type == "xgb_regression":
    config = Epoch.FILE.get_xgb_regression_training_config()
else:
    raise ValueError(f"The provided type of model could not be processed.")



# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the highest lookback.
Candlestick.init(max([m["lookback"] for m in config["models"]]), Epoch.START, Epoch.END)



# CERTIFICATES
# This file will be used to create the certificates batch once all models finish training.
certificates: Union[List[IKerasRegressionTrainingCertificate], List[IXGBRegressionTrainingCertificate]] = []



# TRAINER INSTANCE
# Returns the instance of a training class based on the selected model_type
def get_trainer(
    model_config: Union[IKerasRegressionTrainingConfig, IXGBRegressionTrainingConfig]
) -> Union[KerasRegressionTraining, XGBRegressionTraining]:
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
    trainer: Union[KerasRegressionTraining, XGBRegressionTraining] = get_trainer(model_config)

    # Log the progress
    if index == 0:
        print("\nREGRESSION TRAINING RUNNING\n")
        print(f"{config['name']}\n")
    print(f"\n{index + 1}/{len(config['models'])}) {Utils.prettify_model_id(model_config['id'])}")

    # Train the model
    cert: Union[IKerasRegressionTrainingCertificate, IXGBRegressionTrainingCertificate] = trainer.train()

    # Add the certificate to the list
    certificates.append(cert)



# CERTIFICATE BATCH SAVING
Epoch.FILE.save_training_certificate_batch(model_type, config["name"], certificates)



# MOVING MODELS TO BANK
Epoch.FILE.move_trained_models_to_bank(model_type, certificates)


print("\n\nREGRESSION TRAINING COMPLETED")