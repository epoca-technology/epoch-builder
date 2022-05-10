from json import load
from modules.regression import IRegressionTrainingConfig, RegressionTraining


# REGRESSION TRAINING
# A RegressionTraining instance can initialize, train and save a forecasting model based on the 
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
#    keras_model: The configuration to be used to initialize a Keras Model
#
#
# REGRESSION TRAINING PROCESS
# The Regression Training Instance will initialize all the required properties, build a model and
# train it. Once the process completes, the model and the training certificate will be saved in the
# {OUTPUT_PATH}/{MODEL_ID}.


# REGRESSION TRAINING CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config_file = open('config/RegressionTraining.json')
config: IRegressionTrainingConfig = load(config_file)


# REGRESSION TRAINING INSTANCE
# The Instance of the Forecasting Training that will be executed
regression_training: RegressionTraining = RegressionTraining(config)


# REGRESSION TRAINING EXECUTION
# Trains the regression model based on the provided configuration.
# Results as saved when the execution has completed. If the execution is interrupted before
# completion, results will not be saved.
print("REGRESSION TRAINING RUNNING\n")
regression_training.run()
print("\nREGRESSION TRAINING COMPLETED")
