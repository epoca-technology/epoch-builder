from typing import TypedDict, Union, List, Literal
from modules.types.model_types import IPrediction, IModel




## Position ##


# Types of positions
IPositionType = Literal[1, -1]


# Position Record
# When a position is closed, it is saved in a list that can be reviewed in the GUI when
# the backtest completes.
class IBacktestPosition(TypedDict):
    # Type of position: 1 = long, -1 = short
    t: IPositionType

    # Prediction Dict
    p: IPrediction

    # Position Times
    ot: int                 # Open Timestamp
    ct: Union[int, None]    # Close Timestamp - Populated when the position is closed

    # Position Prices
    op: float   # Open Price
    tpp: float  # Take Profit Price
    slp: float  # Stop Loss Price

    # Close Price: This property is populated when a position is closed. It will
    # take value of the Take Profit Price or Stop Loss Price depending on the outcome.
    cp: Union[float, None]  
    
    # The outcome is populated once the position is closed. True for successful and False
    # for unsuccessful
    o: Union[bool, None]

    # Points when the position is closed
    pts: Union[float, None]




# Performance
# Once a model has finished the testing process it builds a performance dict 
# containing all the details.
class IBacktestPerformance(TypedDict):
    # Points
    points: float               # Total Points Accumulated
    points_hist: List[float]    # Historical fluctuation of points
    points_median: float        # The median of the points collected during the backtest

    # Positions List
    positions: List[IBacktestPosition]

    # Counts
    long_num: int     # Number of closed long positions
    short_num: int    # Number of closed short positions

    # Outcome Counts
    long_outcome_num: int   # Number of price increase outcomes
    short_outcome_num: int  # Number of price decrease outcomes

    # Accuracy
    long_acc: float     # Longs Accuracy
    short_acc: float    # Shorts Accuracy
    general_acc: float  # General Accuracy









## Backtest ##



# Backtest Identifier
# In order to automate several processes, Backtest IDs should be recognizable on a
# file system level. Eventhough the type appears strict, the system should be able to
# handle any string as an ID.
IBacktestID = Literal[
    # Unit Test
    "unit_test",

    # Arima Backtests
    "arima_1", "arima_2", "arima_3", "arima_4", "arima_5", "arima_6", "arima_7", "arima_8", "arima_9",

    # Keras Regression Backtests
    "keras_regression",

    # XGBoost Regression Backtests
    "xgb_regression",

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

    # The list of Model & MultiModel instances that will be put through the backtesting process
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
    model_duration: int     # The model backtesting duration in minutes

    



# Model Backtest Result
# These results are saved by model. The result file which is generated at the end
# of the execution, contains a List[IBacktestResult]
class IBacktestResult(TypedDict):
    backtest: IBacktest
    model: IModel
    performance: IBacktestPerformance