from typing import TypedDict, Union, List
from modules.model import SingleModel, MultiModel, IPrediction, IModel



## Position ##


# Position Record
# When a position is closed, it is saved in a list that can be reviewed in the GUI when
# the backtest completes.
class IBacktestPosition(TypedDict):
    # Type of position: 1 = long, -1 = short
    t: int

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
    points: float            # Total Points Accumulated
    points_hist: List[float] # Historical fluctuation of points

    # Positions List
    positions: List[IBacktestPosition]

    # Counts
    long_num: int     # Number of closed long positions
    short_num: int    # Number of closed short positions

    # Accuracy
    long_acc: float     # Longs Accuracy
    short_acc: float    # Shorts Accuracy
    general_acc: float  # General Accuracy









## Backtest ##




# Backtest Configuration
# A backtest instance can be initialized with the following configuration. Different
# configurations can be spread among multiple backtest instances.
class IBacktestConfig(TypedDict):
    # Identification
    id: str

    # Description of the backtest (purpose)
    description: str

    # Start and end time - If none provided, will use all the available data
    start: Union[str, int, None]
    end: Union[str, int, None]

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
    id: str
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