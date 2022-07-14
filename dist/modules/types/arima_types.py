from typing import TypedDict, Union, Literal




# Arima Config Value
IArimaConfigValue = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# Arima Config
# The configuration to be used on the Arima Instance to generate predictions.
class IArimaConfig(TypedDict):
    # p is the order (number of time lags) of the autoregressive model
    # d is the degree of differencing (the number of times the data have had past values subtracted)
    # q is the order of the moving-average model.
    p: IArimaConfigValue              
    d: IArimaConfigValue              
    q: IArimaConfigValue

    # P, D, Q refer to the autoregressive, differencing, and moving average terms for the 
    # seasonal part of the ARIMA model.
    P: Union[IArimaConfigValue, None]
    D: Union[IArimaConfigValue, None]
    Q: Union[IArimaConfigValue, None]

    # m refers to the number of periods in each season
    m: Union[IArimaConfigValue, None]


