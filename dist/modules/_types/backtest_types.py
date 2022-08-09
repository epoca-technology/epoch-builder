from typing import TypedDict, List, Literal
from modules._types.model_types import IModel
from modules._types.position_types import IPositionPerformance




# Backtest Identifier
# In order to automate several processes, Backtest IDs should be recognizable on a
# file system level. Eventhough the type appears strict, the system should be able to
# handle any string as an ID.
IBacktestID = Literal[
    # Unit Test
    "unit_test",

    # Keras Classification Backtests
    "keras_classification",

    # XGBoost Classification Backtests
    "xgb_classification",

    # Final Shortlist - Includes best classification and consensus models
    "final"
]






# Backtest Configuration
# A backtest instance can be initialized with the following configuration. Different
# configurations can be spread among multiple backtest instances.
class IBacktestConfig(TypedDict):
    # Identification
    id: IBacktestID

    # Description of the backtest (purpose)
    description: str

    # Postitions Take Profit & Stop Loss
    take_profit: float
    stop_loss: float

    # The number of minutes the model will remain idle after closing a position
    idle_minutes_on_position_close: int

    # The list of Model instances that will be put through the backtesting process
    models: List[IModel]




# Backtest
# The backtest configuration summary that is inserted into each of the Model Backtest Results
class IBacktest(TypedDict):
    id: IBacktestID
    description: str
    start: int  # The first candlestick's open time
    end: int    # The last candlestick's close time
    take_profit: float
    stop_loss: float
    idle_minutes_on_position_close: int
    model_start: int        # The time in which the model started backtesting
    model_end: int          # The time in which the model ended backtesting

    



# Model Backtest Result
# These results are saved by model. The result file which is generated at the end
# of the execution, contains a List[IBacktestResult]
class IBacktestResult(TypedDict):
    backtest: IBacktest
    model: IModel
    performance: IPositionPerformance