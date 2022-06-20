from modules.types import IBacktestPath



## Backtest Path Dict ##
BACKTEST_PATH: IBacktestPath = {
    # Backtest Assets
    # The root path for the assets
    "assets": "backtest_assets",

    # Backtest Results
    # The path in which all backtest results are stored. These files should be moved 
    # to final_results once the process completes.
    "results": "backtest_assets/results",

    # Regression Selections
    # The path in which all regression selection results are stored.
    "regression_selections": "backtest_assets/regression_selections",
}