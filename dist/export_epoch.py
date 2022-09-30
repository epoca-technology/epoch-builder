from typing import List
from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch



# EXPORT EPOCH
# Args:
#   --model_ids "_ALPHA_154773ae-aa9b-4dc1-9c48-bd7be1289209,_ALPHA_ef6b3758-9328-4598-b874-0e60292fcbc9,_ALPHA_f2c10b00..."
endpoint_name: str = "EXPORT EPOCH"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--model_ids", dest="model_ids")
args = parser.parse_args()
model_ids: List[str] = args.model_ids.split(",")



# Initialize the Epoch
Epoch.init()



# Export the Epoch
Epoch.export(model_ids)



# End of Script
Utils.endpoint_footer(endpoint_name)