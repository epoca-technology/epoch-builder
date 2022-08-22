from typing import List, Dict
from inquirer import List as InquirerList, prompt
from modules._types import IBacktestConfig
from modules.epoch.Epoch import Epoch
from modules.backtest.Backtest import Backtest



# EPOCH INIT
Epoch.init()


## WELCOME ##
print("BACKTEST\n")


# BACKTEST CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
print(" ")
backtest_config_names: List[str] = Epoch.FILE.list_backtest_configs()
if len(backtest_config_names) == 0:
    raise RuntimeError("The backtest configurations directory is empty.")
backtest_config: Dict[str, str] = prompt([InquirerList("name", message="Select the Backtest Configuration", choices=backtest_config_names)])
config: IBacktestConfig = Epoch.FILE.get_backtest_config(backtest_config["name"])



# BACKTEST INSTANCE
# The Instance of the Backtest that will be executed
backtest: Backtest = Backtest(config)



# BACKTEST EXECUTION
# Runs Backtests for each model, 1 by 1 as well as displaying the progress per instance. 
# Results as saved when the Backtest Instances completes. If the test is interrupted before
# completion, results will not be saved.
print("BACKTEST RUNNING\n")
print(f"{backtest.id}:\n")
backtest.run()
print("\n\nBACKTEST COMPLETED")
