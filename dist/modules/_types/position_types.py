from typing import TypedDict, Union, List, Literal
from modules._types.prediction_types import IPrediction



# Types of positions
IPositionType = Literal[1, -1]




# Position Record
# When a position is closed, it is saved in a list that can be reviewed in the GUI when
# the backtest completes.
class IPosition(TypedDict):
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
class IPositionPerformance(TypedDict):
    # Points
    points: float               # Total Points Accumulated
    points_hist: List[float]    # Historical fluctuation of points
    points_median: float        # The median of the points collected during the backtest

    # Positions List
    positions: List[IPosition]

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