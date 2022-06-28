from typing import TypedDict
from modules.types.model_types import IPrediction



# Regression Prediction Record
# The complete record stored in the database.
class IRegressionPredictionRecord(TypedDict):
    # The identifier of the RegressionModel. F.e: SOME_REGRESSION_ID
    id: str

    # The first open time of the lookback candlesticks in milliseconds.
    fot: int

    # The last close time of the lookback candlesticks in milliseconds.
    lct: int

    # The number of predictions the RegressionModel Outputs
    pn: int

    # The long percentage set in the interpreter.
    l: float

    # The short percentage set in the interpreter.
    s: float

    # The prediction's dictionary
    p: IPrediction

    




# Classification Prediction Record
# The complete record stored in the database.
class IClassificationPredictionRecord(TypedDict):
    # The identifier of the RegressionModel. F.e: SOME_REGRESSION_ID
    id: str

    # The first open time of the lookback candlesticks in milliseconds.
    fot: int

    # The last close time of the lookback candlesticks in milliseconds.
    lct: int

    # The minimum probability set in the interpreter.
    mp: float

    # The prediction's dictionary
    p: IPrediction
