from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch



# BUILD PREDICTION MODELS
# Args:
#   --regression_ids "KR_LSTM_S2_c064c7c8-9208-472a-b963-007225372c08,KR_CDNN_S4_0c5cd87e-b71d-409a-887a-4cd12a8bf6ee,..."
#   --max_combinations? "10000"
endpoint_name: str = "BUILD PREDICTION MODELS"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--regression_ids", dest="action")
parser.add_argument("--max_combinations", dest="id", nargs='?')
args = parser.parse_args()



# Generate the assets (if applies)
# @TODO



# Build the prediction models
# @TODO





# End of Script
Utils.endpoint_footer(endpoint_name)