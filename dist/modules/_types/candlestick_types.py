from typing import Tuple, TypedDict



# Candlesticks Config Type
class ICandlestickConfig(TypedDict):
    columns: Tuple[str]
    csv_file: str
    interval_minutes: int



# Candlestick Build Payload
# This dict is generated when the candlesticks are built during the Epoch Creation Process.
class ICandlestickBuildPayload(TypedDict):
    # Date range of the Epoch
    start: int
    end: int

    # Date range of the Test Dataset
    test_ds_start: int
    test_ds_end: int

    # Highest and Lowest Price SMA
    highest_price_sma: float
    lowest_price_sma: float




# Candlestick Dict
# When iterating over candlesticks, it is more efficient to turn the df into a list 
# of records.
class ICandlestick(TypedDict):
    ot: int     # Open Time
    ct: int     # Close Time
    o: float    # Open Price
    h: float    # High Price
    l: float    # Low Price
    c: float    # Close Price
    v: float    # Volume (USDT)