from argparse import ArgumentParser
from modules._types import IBacktestConfig
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.backtest.Backtest import Backtest




# BACKTEST
# Args:
#   --config_file_name "unit_test.json"
endpoint_name: str = "BACKTEST"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)



# EPOCH INIT
Epoch.init()



# Extract the args
parser = ArgumentParser()
parser.add_argument("--config_file_name", dest="config_file_name")
args = parser.parse_args()



# BACKTEST CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config: IBacktestConfig = Epoch.FILE.get_backtest_config(args.config_file_name)



# BACKTEST INSTANCE
# The Instance of the Backtest that will be executed
backtest: Backtest = Backtest(config)



# BACKTEST EXECUTION
# Runs Backtests for each model, 1 by 1 as well as displaying the progress per instance. 
# Results as saved when the Backtest Instances completes. If the test is interrupted before
# completion, results will not be saved.
print(f"Running: {backtest.id}:\n")
backtest.run()



# End of Script
Utils.endpoint_footer(endpoint_name)
