from modules.types import IBacktestConfig
from modules.epoch.Epoch import Epoch
from modules.epoch.PositionExitCombination import PositionExitCombination
from modules.backtest.Backtest import Backtest



# EPOCH INIT
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
print("BACKTEST RUNNING\n")
print(f"{backtest.id} ({PositionExitCombination.get_id(backtest.take_profit, backtest.stop_loss)}):\n")
backtest.run()
print("\n\nBACKTEST COMPLETED")
