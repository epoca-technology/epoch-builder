from typing import TypedDict, Union




# Arima Config
# The configuration to be used on the Arima Instance to generate predictions.
class IArimaConfig(TypedDict):
    # p is the order (number of time lags) of the autoregressive model
    # d is the degree of differencing (the number of times the data have had past values subtracted)
    # q is the order of the moving-average model.
    p: int              
    d: int              
    q: int

    # P, D, Q refer to the autoregressive, differencing, and moving average terms for the 
    # seasonal part of the ARIMA model.
    P: Union[int, None]
    D: Union[int, None]
    Q: Union[int, None]

    # m refers to the number of periods in each season
    m: Union[int, None]


