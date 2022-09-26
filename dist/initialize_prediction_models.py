from typing import List
from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.prediction_model.PredictionModelAssets import PredictionModelAssets
from modules.prediction_model.PredictionModelConfig import PredictionModelConfig


# INITIALIZE PREDICTION MODELS
# Args:
#   --regression_ids "KR_LSTM_S2_c064c7c8-9208-472a-b963-007225372c08,KR_CDNN_S4_0c5cd87e-b71d-409a-887a-4cd12a8bf6ee,..."
endpoint_name: str = "INITIALIZE PREDICTION MODELS"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--regression_ids", dest="regression_ids")
args = parser.parse_args()
regression_ids: List[str] = args.regression_ids.split(",")


# Initialize the Epoch
Epoch.init()


# Initialize the Candlesticks on the Test Dataset Range
Candlestick.init(Epoch.REGRESSION_LOOKBACK, Epoch.TEST_DS_START, Epoch.TEST_DS_END)



# Build the assets
PredictionModelAssets.build(regression_ids, PredictionModelConfig.PRICE_CHANGE_REQUIREMENTS)



# Create the configs
PredictionModelConfig.create(regression_ids)



# End of Script
Utils.endpoint_footer(endpoint_name)