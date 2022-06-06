from typing import List
from modules.database import IDatabaseTable






## Tables SQL Queries ##



# Arima Predictions Table
# The table in which the Arima Predictions are stored.
# Column Descriptions:
# id: The identifier of the ArimaModel. F.e: A565
# fot: The first open time of the lookback candlesticks in milliseconds.
# lct: The last close time of the lookback candlesticks in milliseconds.
# pn: The number of predictions the ArimaModel Outputs
# l: The long percentage set in the interpreter.
# s: The short percentage set in the interpreter.
# p: The prediction's dictionary
# Notice there is no need to store the lookback value as it can be derived from
# the fot and lct.
def ARIMA_PREDICTIONS_TABLE_SQL(table_name: str) -> str:
    """Returns the Arima Predictions Table's SQL based on the provided table name.

    Args:
        table_name: str
            When initializing the table, make sure to also initialize it with
            the test_ preffix.
    
    Returns: 
        str
    """
    return f"\
            CREATE TABLE IF NOT EXISTS {table_name}(\
                id  VARCHAR(1000) NOT NULL,\
                fot BIGINT NOT NULL,\
                lct BIGINT NOT NULL,\
                pn  SMALLINT NOT NULL,\
                l   REAL NOT NULL,\
                s   REAL NOT NULL,\
                p   JSONB NOT NULL\
            );\
            CREATE INDEX IF NOT EXISTS {table_name}_id ON {table_name}(id);\
            CREATE INDEX IF NOT EXISTS {table_name}_fot ON {table_name}(fot);\
            CREATE INDEX IF NOT EXISTS {table_name}_lct ON {table_name}(lct);\
            CREATE INDEX IF NOT EXISTS {table_name}_pn ON {table_name}(pn);\
            CREATE INDEX IF NOT EXISTS {table_name}_l ON {table_name}(l);\
            CREATE INDEX IF NOT EXISTS {table_name}_s ON {table_name}(s);\
        "





# Regression Predictions Table
# The table in which the Regression Predictions are stored.
# Column Descriptions:
# id: The identifier of the RegressionModel. F.e: SOME_REGRESSION_ID
# fot: The first open time of the lookback candlesticks in milliseconds.
# lct: The last close time of the lookback candlesticks in milliseconds.
# pn: The number of predictions the RegressionModel Outputs
# l: The long percentage set in the interpreter.
# s: The short percentage set in the interpreter.
# p: The prediction's dictionary
# Notice there is no need to store the lookback value as it can be derived from
# the fot and lct.
def REGRESSION_PREDICTIONS_TABLE_SQL(table_name: str) -> str:
    """Returns the Regression Predictions Table's SQL based on the provided table name.

    Args:
        table_name: str
            When initializing the table, make sure to also initialize it with
            the test_ preffix.
    
    Returns: 
        str
    """
    return f"\
            CREATE TABLE IF NOT EXISTS {table_name}(\
                id  VARCHAR(1000) NOT NULL,\
                fot BIGINT NOT NULL,\
                lct BIGINT NOT NULL,\
                pn  SMALLINT NOT NULL,\
                l   REAL NOT NULL,\
                s   REAL NOT NULL,\
                p   JSONB NOT NULL\
            );\
            CREATE INDEX IF NOT EXISTS {table_name}_id ON {table_name}(id);\
            CREATE INDEX IF NOT EXISTS {table_name}_fot ON {table_name}(fot);\
            CREATE INDEX IF NOT EXISTS {table_name}_lct ON {table_name}(lct);\
            CREATE INDEX IF NOT EXISTS {table_name}_pn ON {table_name}(pn);\
            CREATE INDEX IF NOT EXISTS {table_name}_l ON {table_name}(l);\
            CREATE INDEX IF NOT EXISTS {table_name}_s ON {table_name}(s);\
        "





# Classification Predictions
# @TODO







# Technical Analysis Table
# The table in which the Technical Analysis data is stored.
# Column Descriptions:
# id: The identifier of the range that is being covered. F.e: FIRSTOT_LASTCT
# ta: The technical analysis dictionary
def TECHNICAL_ANALYSIS_TABLE_SQL(table_name: str) -> str:
    """Returns the Technical Analysis Table's SQL based on the provided table name.

    Args:
        table_name: str
            When initializing the table, make sure to also initialize it with
            the test_ preffix.
    
    Returns: 
        str
    """
    return f"\
            CREATE TABLE IF NOT EXISTS {table_name}(\
                id  VARCHAR(150) NOT NULL PRIMARY KEY,\
                ta  JSONB NOT NULL\
            );\
        "











## Tables List ##
# The list of tables that is ready to be initialized.
TABLES: List[IDatabaseTable] = [
    {
        "name": "arima_predictions",
        "sql": ARIMA_PREDICTIONS_TABLE_SQL
    },
    {
        "name": "regression_predictions",
        "sql": REGRESSION_PREDICTIONS_TABLE_SQL
    },
    {
        "name": "technical_analysis",
        "sql": TECHNICAL_ANALYSIS_TABLE_SQL
    }
]


