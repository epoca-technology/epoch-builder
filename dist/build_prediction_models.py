from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.prediction_model.PredictionModel import PredictionModel


# BUILD PREDICTION MODELS
# Args:
#   --limit "100"
endpoint_name: str = "BUILD PREDICTION MODELS"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--limit", dest="limit")
args = parser.parse_args()


# Initialize the Epoch
Epoch.init()


# Initialize the Candlesticks on the Test Dataset Range
Candlestick.init(Epoch.REGRESSION_LOOKBACK, Epoch.TEST_DS_START, Epoch.TEST_DS_END)



# Initialize the instance of the builder
PredictionModel().build(int(args.limit))



# End of Script
Utils.endpoint_footer(endpoint_name)