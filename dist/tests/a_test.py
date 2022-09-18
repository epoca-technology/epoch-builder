from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch

# Initialize the Epoch
Epoch.init()

# Initialize the candlesticks
Candlestick.init(Epoch.REGRESSION_LOOKBACK, Epoch.START, Epoch.END)