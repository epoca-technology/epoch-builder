from typing import List, TypedDict, Union
from pandas import DataFrame, Series, read_csv
from ta.momentum import rsi
from ta.trend import aroon_up, aroon_down
from modules.utils import Utils





# Type
class ITechnicalAnalysis(TypedDict):
    rsi: Union[float, None]
    aroon_up: Union[float, None]
    aroon_down: Union[float, None]





# Class
class TechnicalAnalysis:
    """TechnicalAnalysis Class

    This singleton manages the technical analysis indicators for Classifications.


    Class Properties:
        SHORT_WINDOW: int 
        MEDIUM_WINDOW: int 
        LONG_WINDOW: int 
            The windows that will be use for indicators.
        DECIMALS: int
            The number of decimals that will be used in the normalization.
        
    """
    # Windows
    SHORT_WINDOW: int = 7
    MEDIUM_WINDOW: int = 25
    LONG_WINDOW: int = 50

    # Normalization Decimals
    DECIMALS: int = 6






    @staticmethod
    def get_technical_analysis(
        lookback_df: DataFrame, 
        include_rsi: bool=False, 
        include_aroon: bool=False
    ) -> ITechnicalAnalysis:
        """Given a date_value, it will process it according to its format and return the 
        equivalent timestamp in milliseconds.

        Args:
            date_value: Union[str, int, None] 
                The date that needs to be converted.

        Returns:
            ITechnicalAnalysis
        """
        pass
















    ## Misc Helpers ##





    @staticmethod
    def _get_id(first_ot: int, last_ct: int) -> str:
        """Given a lookback range, it will turn it into an id that's compatible with the
        cache system.

        Args:
            first_ot: int
                The open time of the first candlestick of the lookback.
            first_ot: int
                The close time of the last candlestick of the lookback.

        Returns:
            str
        """
        return f"{str(int(first_ot))}_{str(int(last_ct))}"









    @staticmethod
    def _normalize_value(value: float) -> float:
        """Normalizes given value by dividing it by 100.

        Args:
            value: float
                The value to be normalized

        Returns:
            float
        """
        return round(value / 100, TechnicalAnalysis.DECIMALS)