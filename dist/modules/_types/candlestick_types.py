from typing import Tuple, TypedDict, Dict, List



# Lookback Prediction Range Indexer
IPredictionRangeIndexer = Dict[str, List[int]]



# Candlesticks Config Type
class ICandlestickConfig(TypedDict):
    columns: Tuple[str]
    csv_file: str
    interval_minutes: int