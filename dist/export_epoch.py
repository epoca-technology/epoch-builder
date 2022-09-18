from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch



# EXPORT EPOCH
# Args:
#   --model_id "c064c7c8-9208-472a-b963-007225372c08"
endpoint_name: str = "EXPORT EPOCH"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# Extract the args
parser = ArgumentParser()
parser.add_argument("--model_id", dest="model_id")
args = parser.parse_args()



# Export the Epoch
Epoch.export(args.model_id)



# End of Script
Utils.endpoint_footer(endpoint_name)