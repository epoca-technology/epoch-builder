from typing import List
from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch



# EXPORT EPOCH
# Args:
#   --model_id "_ALPHA_154773ae-aa9b-4dc1-9c48-bd7be1289209"
endpoint_name: str = "EXPORT EPOCH"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--model_id", dest="model_id")
args = parser.parse_args()



# Initialize the Epoch
Epoch.init()



# Export the Epoch
Epoch.export(args.model_id)



# End of Script
Utils.endpoint_footer(endpoint_name)