from typing import List
from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.regression_selection.RegressionSelection import RegressionSelection


# REGRESSION SELECTION
# Args:
#   --model_ids "KR_LSTM_S2_c064c7c8-9208-472a-b963-007225372c08,KR_LSTM_S3_8e00946d-dd77-46eb-bb3a-e5e92369ad7a,..."
endpoint_name: str = "REGRESSION SELECTION"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# EPOCH INIT
Epoch.init()



# Extract the args
parser = ArgumentParser()
parser.add_argument("--model_ids", dest="model_ids")
args = parser.parse_args()



# Init the Model IDs
model_ids: List[str] = args.model_ids.split(",")



# Initialize the instance and execute the selection
rs: RegressionSelection = RegressionSelection(model_ids)
rs.run()



# End of Script
Utils.endpoint_footer(endpoint_name)