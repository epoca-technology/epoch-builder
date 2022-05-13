from typing import List
from os import makedirs
from os.path import exists
from json import load, dumps
from modules.utils import Utils
from modules.candlestick import Candlestick
from modules.keras_models import KERAS_MODELS_CERTIFICATES_PATH
from modules.regression import IRegressionTrainingBatch, RegressionTraining, IRegressionTrainingCertificate


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




# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the highest lookback among the models config.
Candlestick.init(max([m["lookback"] for m in config["models"]]), normalized_df=True)




# REGRESSION TRAINING EXECUTION
# Builds, trains and saves all the models as well as their certificates. Once all models have been 
# trained, saves the certificates batch. 
# Results as saved when the execution has completed. If the execution is interrupted before
# completion, results will not be saved.


# Init the list of certificates
certificates: List[IRegressionTrainingCertificate] = []

# Run the training
for index, model_config in enumerate(config["models"]):
    # Initialize the instance of the model
    regression_training: RegressionTraining = RegressionTraining(model_config)

    # Print the progress
    if index == 0:
        print("REGRESSION TRAINING RUNNING\n")
    print(f"{index + 1}/{len(config['models'])}) {model_config['id']}")

    # Train the model
    cert: IRegressionTrainingCertificate = regression_training.train()

    # Add the certificate to the list
    certificates.append(cert)

# Save the certificates
if not exists(KERAS_MODELS_CERTIFICATES_PATH):
    makedirs(KERAS_MODELS_CERTIFICATES_PATH)
with open(f"{KERAS_MODELS_CERTIFICATES_PATH}/{config['name']}_{Utils.get_time()}.json", "w") as outfile:
    outfile.write(dumps(certificates, indent=4))
print("\nREGRESSION TRAINING COMPLETED")