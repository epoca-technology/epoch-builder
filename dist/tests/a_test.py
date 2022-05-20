from modules.candlestick import Candlestick
from modules.model import ArimaModel
from modules.database import Database

# Set the Database on test mode
Database.TEST_MODE = True

# Initialize the candlesticks
Candlestick.init(ArimaModel.DEFAULT_LOOKBACK)

# Notify the user
print("PREDICTION BACKTESTING UNIT TESTS\n")