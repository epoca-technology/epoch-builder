from modules.candlestick.Candlestick import Candlestick
from modules.database.Database import Database
from modules.epoch.Epoch import Epoch

# Set the Database on test mode
Database.TEST_MODE = True

# Initialize the Epoch
Epoch.init()

# Initialize the candlesticks
Candlestick.init(300)

# Notify the user
print("PREDICTION BACKTESTING UNIT TESTS\n")