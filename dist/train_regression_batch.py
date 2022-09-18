from typing import List, Union
from argparse import ArgumentParser
from modules._types import IRegressionTrainingConfigBatch, IRegressionTrainingCertificate, \
    IRegressionTrainAndTestDatasets
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.regression.RegressionTrainingConfig import RegressionTrainingConfig
from modules.regression.RegressionTraining import RegressionTraining



# REGRESSION TRAINING
# Args:
#   --category "DNN"
#   --batch_file_name "KR_LSTM_7_23.json"
endpoint_name: str = "REGRESSION TRAINING"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)




# EPOCH INIT
Epoch.init()




# Extract the args
parser = ArgumentParser()
parser.add_argument("--category", dest="category")
parser.add_argument("--batch_file_name", dest="batch_file_name")
args = parser.parse_args()



# CANDLESTICK INITIALIZATION
# Initialize the Candlesticks Module based on the regression lookback.
Candlestick.init(Epoch.REGRESSION_LOOKBACK, Epoch.START, Epoch.END)




# Retrieve the batch config
config_batch: IRegressionTrainingConfigBatch = RegressionTrainingConfig.get_batch(args.category, args.batch_file_name)



# DATASETS
# The packed datasets that will be used to train and evaluate regressions.
datasets: IRegressionTrainAndTestDatasets = RegressionTraining.make_train_and_test_datasets()




# CERTIFICATES
# This file will be used to create the certificates batch once all models finish training.
certificates: List[IRegressionTrainingCertificate] = []




# REGRESSION TRAINING EXECUTION
# Builds, trains, saves and evaluates models. Once models finish training, their file as well as the
# certificate is saved in the models directory. 
# When the process completes, it saves all the certificates into a single batch.
# Keep in mind that the training of a model will be skipped if the certificate exists.
print(f"Batch: {config_batch['name']}\n")
for index, model_config in enumerate(config_batch["configs"]):
    # Retrieve the certificate (if any)
    cert: Union[IRegressionTrainingCertificate, None] = RegressionTraining.get_certificate(model_config["id"])

    # Check if the model needs to be trained
    if cert is None:
        # Log the progress
        print(f"\n{index + 1}/{len(config_batch['configs'])}) {model_config['id']}")

        # Initialize the training instance
        training: RegressionTraining = RegressionTraining(model_config, datasets)

        # Train and discover the model
        cert = training.train()
    
    # Otherwise, skip the training
    else:
        # Log the progress
        print(f"\n{index + 1}/{len(config_batch['configs'])}) {model_config['id']}: Skipped")

    # Add the certificate to the list
    certificates.append(cert)



# Save the batched training certificates once they are all available
RegressionTraining.save_certificates_batch(config_batch["name"], certificates)






# End of Script
Utils.endpoint_footer(endpoint_name)