from typing import TypedDict, Union



# ARIMA COMBINATIONS CONFIGURATION
# Based on the values provided in the configuration files, the appropiate backtest files will
# be generated and placed in the output directory.
class IArimaCombinationsConfig(TypedDict):
    # Base ID. This value will be changed when batching
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

    # The number that will be focused when generating combinations for p,d and q
    focus_number: int

    # The maximum number of models that should go in each file
    batch_size: int





# ARIMA COMBINATION
# The dictionary that contains a single Arima Combination (p, d, q).
class IArimaCombination(TypedDict):
    p: int
    d: int
    q: int