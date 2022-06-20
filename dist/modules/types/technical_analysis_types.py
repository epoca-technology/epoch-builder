from typing import TypedDict, Union




class ITechnicalAnalysis(TypedDict):
    rsi: Union[float, None]
    stoch: Union[float, None]
    aroon: Union[float, None]
    stc: Union[float, None]
    mfi: Union[float, None]