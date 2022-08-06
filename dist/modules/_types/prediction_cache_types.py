from typing import TypedDict
from modules._types.prediction_types import IPrediction



# Prediction Record
# The complete record stored in the database.
class ICachedPredictionRecord(TypedDict):
    # The identifier of the Model.
    id: str

    # The first open time of the lookback candlesticks in milliseconds.
    fot: int

    # The last close time of the lookback candlesticks in milliseconds.
    lct: int

    # The prediction's dictionary
    p: IPrediction

    




# Feature Record
# The complete record stored in the database.
class ICachedFeatureRecord(TypedDict):
    # The identifier of the Model.
    id: str

    # The first open time of the lookback candlesticks in milliseconds.
    fot: int

    # The last close time of the lookback candlesticks in milliseconds.
    lct: int

    # The feature's value
    f: float
