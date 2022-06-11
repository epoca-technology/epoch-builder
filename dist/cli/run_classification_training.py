from typing import List, Dict, Union
from os import makedirs, remove
from os.path import exists
from json import load, dumps
from inquirer import Text, prompt
from tqdm import tqdm
from modules.utils import Utils
from modules.keras_models import KERAS_PATH
from modules.classification import IClassificationTrainingBatch, ClassificationTraining, IClassificationTrainingCertificate, \
    ITrainingDataFile


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
# Init the max evaluations
answers: Dict[str, str] = prompt([Text("max_evaluations", "Number of Classification Evaluations (Defaults to 250)")])
max_evaluations: Union[int, None] = int(answers["max_evaluations"]) if answers["max_evaluations"].isdigit() else None

# Init the list of certificates
certificates: List[IClassificationTrainingCertificate] = []

# Run the training
for index, model_config in enumerate(config["models"]):
    # Initialize the instance of the model
    classification_training: ClassificationTraining = ClassificationTraining(
        training_data, 
        model_config, 
        max_evaluations=max_evaluations,
        hyperparams_mode=config["hyperparams_mode"]
    )

    # Log the progress
    if index == 0:
        print("\nCLASSIFICATION TRAINING RUNNING")
        if config["hyperparams_mode"]:
            print("\n")
            progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=len(config["models"]))

    if not config["hyperparams_mode"]:    
        print(f"\n{index + 1}/{len(config['models'])}) {model_config['id']}")

    # Train the model
    cert: IClassificationTrainingCertificate = classification_training.train()

    # Add the certificate to the list
    certificates.append(cert)

    # Perform the post evaluation cleanup if applies
    if config["hyperparams_mode"]:
        remove(f"{KERAS_PATH['models']}/{classification_training.id}/model.h5")
        progress_bar.update()

# Save the certificates
if not exists(KERAS_PATH["batched_training_certificates"]):
    makedirs(KERAS_PATH["batched_training_certificates"])
with open(f"{KERAS_PATH['batched_training_certificates']}/{config['name']}_{Utils.get_time()}.json", "w") as outfile:
    outfile.write(dumps(certificates))
print("\nCLASSIFICATION TRAINING COMPLETED")