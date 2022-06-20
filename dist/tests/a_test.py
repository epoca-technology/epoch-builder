from modules.candlestick.Candlestick import Candlestick
from modules.database.Database import Database

# Set the Database on test mode
Database.TEST_MODE = True

# Initialize the candlesticks
Candlestick.init(300)

# Notify the user
print("PREDICTION BACKTESTING UNIT TESTS\n")