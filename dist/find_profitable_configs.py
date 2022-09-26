from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.prediction_model.PredictionModel import PredictionModel


# FIND PROFITABLE CONFIGS
# Args:
#   --batch_file_name "_ALPHA_1_10.json"
endpoint_name: str = "FIND PROFITABLE CONFIGS"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--batch_file_name", dest="batch_file_name")
args = parser.parse_args()


# Initialize the Epoch
Epoch.init()


# Initialize the Candlesticks on the Test Dataset Range
Candlestick.init(Epoch.REGRESSION_LOOKBACK, Epoch.TEST_DS_START, Epoch.TEST_DS_END)



# Initialize the instance of the Prediction Model and run the process
PredictionModel().find_profitable_configs(args.batch_file_name)



# End of Script
Utils.endpoint_footer(endpoint_name)