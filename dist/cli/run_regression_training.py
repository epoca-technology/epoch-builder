from typing import List, Tuple
from os import makedirs, remove
from os.path import exists
from numpy import ndarray
from json import load, dumps
from tqdm import tqdm
from modules.types import IRegressionTrainingBatch, IRegressionTrainingCertificate
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.keras_models.KerasPath import KERAS_PATH
from modules.regression.RegressionTraining import RegressionTraining


# REGRESSION TRAINING
# A RegressionTraining instance can initialize, train and save a regression model based on the 
# configuration file.
#
# Identification:
#   id: The ID of the model. Must be descriptive, compatible with filesystems and preffixed with 'R_'
#   description: Any relevant data that should be attached to the trained model.
#
# Training Configuration:
#    lookback: The number of prediction candlesticks that will look into the past in order to make a prediction.
#    predictions: The number of predictions to be generated
#    learning_rate: The learning rate to be used by the optimizer
#    loss: The loss function to be used to train the model
#    metric: The metric function to be used to evaluate the training the model
#    batch_size: The size of the training batches
#    shuffle_data: Wether the train, val and test data should be shuffled for training.
#    keras_model: The configuration to be used to initialize a Keras Model
#
#
# REGRESSION TRAINING PROCESS
# The Regression Training Instance will initialize all the required properties, build a model and
# train it. Once the process completes, the model and the training certificate will be saved in the
# {KERAS_MODELS_PATH}/{MODEL_ID}. Once all the models have been trained and evaluated, the certificates
# batch is saved in {KERAS_MODELS_CERTIFICATES_PATH}/{BATCH_NAME}_{BATCH_TIME}.json



# REGRESSION TRAINING CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config_file = open('config/RegressionTraining.json')
config: IRegressionTrainingBatch = load(config_file)







# REGRESSION TRAINING EXECUTION
# Builds, trains and saves all the models as well as their certificates. Once all models have been 
# trained, saves the certificates batch. 
# Results as saved when the execution has completed. If the execution is interrupted before
# completion, results will not be saved.


# REGRESSION SHARED PROPERTIES
# For performance reasons, the train and test datasets should be generated only once prior to training.
# IMPORTANT: with this change in place, all models must have the same properties.
lookback: int = config["models"][0]["lookback"]
autoregressive: bool = config["models"][0]["autoregressive"]
predictions: int = config["models"][0]["predictions"]


# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the regressions' lookback.
Candlestick.init(lookback, config.get("start"), config.get("end"))


# DATASETS
# Initialize the train and test datasets for the entire batch of models that will be trained.
datasets: Tuple[ndarray, ndarray, ndarray, ndarray] = RegressionTraining.make_datasets(
    lookback=lookback, 
    autoregressive=autoregressive,
    predictions=predictions
)


# Init the list of certificates
certificates: List[IRegressionTrainingCertificate] = []


# Run the training
for index, model_config in enumerate(config["models"]):
    # Initialize the instance of the model
    regression_training: RegressionTraining = RegressionTraining(
        model_config, 
        hyperparams_mode=config["hyperparams_mode"],
        datasets=datasets
    )

    # Print the progress
    if index == 0:
        print("REGRESSION TRAINING RUNNING\n")
        print(f"{config['name']}\n")
        if config["hyperparams_mode"]:
            progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=len(config["models"]))

    if config["hyperparams_mode"]:
        progress_bar.set_description(f"{model_config['id'][0:20]}...")
    else:    
        print(f"\n{index + 1}/{len(config['models'])}) {model_config['id'][0:20]}...")


    # Train the model
    cert: IRegressionTrainingCertificate = regression_training.train()


    # Add the certificate to the list
    certificates.append(cert)

    # Perform the post evaluation cleanup if applies
    if config["hyperparams_mode"]:
        remove(f"{KERAS_PATH['models']}/{regression_training.id}/model.h5")
        progress_bar.update()

# Save the certificates
if not exists(KERAS_PATH["batched_training_certificates"]):
    makedirs(KERAS_PATH["batched_training_certificates"])
with open(f"{KERAS_PATH['batched_training_certificates']}/{config['name']}_{Utils.get_time()}.json", "w") as outfile:
    outfile.write(dumps(certificates))
print("\nREGRESSION TRAINING COMPLETED")