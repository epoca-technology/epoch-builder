from modules.model import Model
from modules.backtest import Backtest


# BACKTEST CONFIGURATION
# A Backtest instance can perform a backtesting process on a series of models.
#
# Identification:
#   id: The identification/description of the Backtest Instance. This value must be compatible
#       with file systems as it will be part of the result name like {BACKTEST_ID}_{TIMESTAMP}.json
# 
# Backtest Date Range:
#   start: The start date for the backtest. It can be a string like '22/01/2021' or a timestamp
#       in milliseconds. If None is provided, it will use all the available data
#   end: The end date for the backtest. It can be a string like '22/01/2021' or a timestamp
#       in milliseconds. If None is provided, it will use all the available data
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
#   models: The list of Model Instances that will be put through the backtest. 
#
# BACKTEST PROCESS
# The Backtest Instance will run the test on the models in order and will output the results to the 
# directory /backtest_results




# BACKTEST INSTANCE
# The Instance of the Backtest that will be executed
backtest: Backtest = Backtest({
    "id": "SOME_BACKTEST_MOCK",
    "description": "This backtest was created in order to be able to test multiple file analysis in the GUI.",
    "start": '1/04/2022',
    "end": None,
    "take_profit": 1,
    "stop_loss": 1,
    "idle_minutes_on_position_close": 30,
    "models": [

        Model({
            'id': 'FIXED_DURATION_01',
            "single_models": [{
                'lookback': 200,
                'arima': { 'predictions': 7, 'p': 2, 'd': 1, 'q': 2 },
                'interpreter': { 'long': 0.05, 'short': 0.05, 'rsi': {'active': True, 'overbought': 70, 'oversold': 30}, 'ema': {'active': True, 'distance': 1} }
            }]
        }),


    ]
})






# BACKTEST EXECUTION
# Runs Backtests for each model, 1 by 1 as well as displaying the progress per instance. 
# Results as saved as the instances complete. If the test is interrupted before
# completion, results will not be saved.
print("PREDICTION BACKTESTING RUNNING")
print(f"\n{backtest.id}:\n")
backtest.run()
print("\nPREDICTION BACKTESTING COMPLETED")
