from typing import List, Tuple
from os import makedirs, remove
from os.path import exists
from shutil import rmtree
from pandas import DataFrame
from json import load, dumps
from tqdm import tqdm
from modules.types import IClassificationTrainingBatch, IClassificationTrainingCertificate, ITrainingDataFile
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.keras_models.KerasPath import KERAS_PATH
from modules.classification.ClassificationTraining import ClassificationTraining
    


# CLASSIFICATION TRAINING
# A ClassificationTraining instance can initialize, train and save a classification model based on the 
# configuration file.
#
# Identification:
#   id: The ID of the model. Must be descriptive, compatible with filesystems and preffixed with 'C_'
#   description: Any relevant data that should be attached to the trained model.
#
# Training Configuration:
#    learning_rate: The learning rate to be used by the optimizer
#    loss: The loss function to be used to train the model
#    metric: The metric function to be used to evaluate the training the model
#    batch_size: The size of the training batches
#    shuffle_data: Wether the train, val and test data should be shuffled for training.
#    keras_model: The configuration to be used to initialize a Keras Model
#
#
# CLASSIFICATION TRAINING PROCESS
# The Classification Training Instance will initialize all the required properties, build a model and
# train it. Once the process completes, the model and the training certificate will be saved in the
# {KERAS_MODELS_PATH}/{MODEL_ID}. Once all the models have been trained and evaluated, the certificates
# batch is saved in {KERAS_MODELS_CERTIFICATES_PATH}/{BATCH_NAME}_{BATCH_TIME}.json




# CLASSIFICATION TRAINING CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config_file = open("config/ClassificationTraining.json")
config: IClassificationTrainingBatch = load(config_file)




# TRAINING DATA FILE
# Opens and loads the training_data file that will be used to train the models.
training_data_config_file = open(f"{KERAS_PATH['classification_training_data']}/{config['training_data_id']}.json")
training_data: ITrainingDataFile = load(training_data_config_file)




# CLASSIFICATION TRAINING EXECUTION
# Builds, trains and saves all the models as well as their certificates. Once all models have been 
# trained, saves the certificates batch. 
# Results as saved when the execution has completed. If the execution is interrupted before
# completion, results will not be saved.
print("CLASSIFICATION TRAINING")


# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the highest lookback among the models config.
Candlestick.init(300, start=training_data["start"], end=training_data["end"])


# DATASETS
# Initialize the train and test datasets for the entire batch of models that will be trained.
datasets: Tuple[DataFrame, DataFrame, DataFrame, DataFrame] = ClassificationTraining.make_datasets(
    training_data=training_data["training_data"]
)


# Init the list of certificates
certificates: List[IClassificationTrainingCertificate] = []


# Run the training
for index, model_config in enumerate(config["models"]):
    # Initialize the instance of the model
    classification_training: ClassificationTraining = ClassificationTraining(
        training_data, 
        model_config, 
        hyperparams_mode=config["hyperparams_mode"],
        datasets=datasets
    )

    # Log the progress
    if index == 0:
        print("\nCLASSIFICATION TRAINING RUNNING\n")
        print(f"{config['name']}\n")
        if config["hyperparams_mode"]:
            progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=len(config["models"]))

    if config["hyperparams_mode"]:
        progress_bar.set_description(f"{model_config['id'][0:20]}...")
    else:    
        print(f"\n{index + 1}/{len(config['models'])}) {model_config['id'][0:20]}...")

    # Train the model
    cert: IClassificationTrainingCertificate = classification_training.train()

    # Add the certificate to the list
    certificates.append(cert)

    # Update progress bar if applies
    if config["hyperparams_mode"]:
        progress_bar.update()

# Save the certificates
if not exists(KERAS_PATH["batched_training_certificates"]):
    makedirs(KERAS_PATH["batched_training_certificates"])
with open(f"{KERAS_PATH['batched_training_certificates']}/{config['name']}_{Utils.get_time()}.json", "w") as outfile:
    outfile.write(dumps(certificates))


# If running on hyperparams mode, clean up the residue
if config["hyperparams_mode"]:
    for cert in certificates:
        rmtree(f"{KERAS_PATH['models']}/{cert['id']}")
print("\nCLASSIFICATION TRAINING COMPLETED")