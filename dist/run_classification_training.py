from typing import List
from os import makedirs
from os.path import exists
from json import load, dumps
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


# Init the list of certificates
certificates: List[IClassificationTrainingCertificate] = []

# Run the training
for index, model_config in enumerate(config["models"]):
    # Initialize the instance of the model
    regression_training: ClassificationTraining = ClassificationTraining(training_data, model_config)

    # Print the progress
    if index == 0:
        print("CLASSIFICATION TRAINING RUNNING")
    print(f"\n{index + 1}/{len(config['models'])}) {model_config['id']}")

    # Train the model
    cert: IClassificationTrainingCertificate = regression_training.train()

    # Add the certificate to the list
    certificates.append(cert)

# Save the certificates
if not exists(KERAS_PATH["batched_training_certificates"]):
    makedirs(KERAS_PATH["batched_training_certificates"])
with open(f"{KERAS_PATH['batched_training_certificates']}/{config['name']}_{Utils.get_time()}.json", "w") as outfile:
    outfile.write(dumps(certificates, indent=4))
print("\nCLASSIFICATION TRAINING COMPLETED")