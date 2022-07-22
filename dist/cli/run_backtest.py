from modules.types import IBacktestConfig
from modules.epoch.Epoch import Epoch
from modules.epoch.PositionExitCombination import PositionExitCombination
from modules.backtest.Backtest import Backtest


# BACKTEST CONFIGURATION FILE
# A Backtest instance can perform a backtesting process on a series of models.
#
# Identification:
#   id: The identification/description of the Backtest Instance. This value must be compatible
#       with file systems as it will be part of the result name like {BACKTEST_ID}_{TIMESTAMP}.json
#
# Positions:
#   take_profit: The percentage that will be applied when opening a position. F.e: If the open price
#       is 100 and the take_profit is set at 10, it will set the take profit price at 110 in the case of a long 
#       or 90 in the case of a short.
#   stop_loss: The percentage that will be applied when opening a position. F.e: If the open price
#       is 100 and the stop_loss is set at 10, it will set the stop loss price at 90 in the case of a long 
#       or 110 in the case of a short.
#   idle_minutes_on_position_close: The number of minutes the model will not trade for after closing a position.
# 
# Models:
#   models: The list of Model Configurations that will be put through the backtest. 
#
# BACKTEST PROCESS
# The Backtest Instance will run the test on the models in order and will output the results to the 
# directory /backtest_assets/results/$position_exit_combination/$backtest_id.json


# Initialize the Epoch
Epoch.init()


# BACKTEST CONFIGURATION
# Opens and loads the configuration file that should be placed in the root of the project.
config: IBacktestConfig = Epoch.FILE.get_backtest_config()


# BACKTEST INSTANCE
# The Instance of the Backtest that will be executed
backtest: Backtest = Backtest(config)


# BACKTEST EXECUTION
# Runs Backtests for each model, 1 by 1 as well as displaying the progress per instance. 
# Results as saved when the Backtest Instances completes. If the test is interrupted before
# completion, results will not be saved.
print("BACKTEST RUNNING")
print(f"\n{backtest.id} ({PositionExitCombination.get_id(backtest.take_profit, backtest.stop_loss)}):\n")
backtest.run()
print("\nBACKTEST COMPLETED")
