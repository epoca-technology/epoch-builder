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
    "id": "APDQ",
    "description": "The purpose of this backtest is to find out how different p, d, q combinations can brin better results.",
    "start": '1/10/2021',
    "end": '22/04/2022',
    "take_profit": 1,
    "stop_loss": 1,
    "idle_minutes_on_position_close": 30,
    "models": [

        Model({
            'id': 'A212',
            "single_models": [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 2, 'd': 1, 'q': 2 },
                'interpreter': { 'long': 0.05, 'short': 0.05 }
            }]
        }),


        Model({
            'id': 'A322',
            "single_models": [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 3, 'd': 2, 'q': 2 },
                'interpreter': { 'long': 0.05, 'short': 0.05 }
            }]
        }),


        Model({
            'id': 'A323',
            "single_models": [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 3, 'd': 2, 'q': 3 },
                'interpreter': { 'long': 0.05, 'short': 0.05 }
            }]
        }),


        Model({
            'id': 'A324',
            "single_models": [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 3, 'd': 2, 'q': 4 },
                'interpreter': { 'long': 0.05, 'short': 0.05 }
            }]
        }),

        Model({
            'id': 'A325',
            "single_models": [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 3, 'd': 2, 'q': 5 },
                'interpreter': { 'long': 0.05, 'short': 0.05 }
            }]
        }),

        Model({
            'id': 'A425',
            "single_models": [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 4, 'd': 2, 'q': 5 },
                'interpreter': { 'long': 0.05, 'short': 0.05 }
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
