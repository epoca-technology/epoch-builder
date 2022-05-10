from modules.candlestick import Candlestick
from modules.model import ArimaModel


# Initialize the candlesticks
Candlestick.init(ArimaModel.DEFAULT_LOOKBACK, normalized_df=True)

# Notify the user
print("PREDICTION BACKTESTING UNIT TESTS\n")