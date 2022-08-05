from typing import List
from modules.types import IDatabaseTable, IDatabaseTableName






## Tables' SQL Queries ##



# Features Table
# The table in which the Regression Features are stored.
# Column Descriptions:
# id: The identifier of the RegressionModel. F.e: KR_SOME_REGRESSION_ID
# fot: The first open time of the lookback candlesticks in milliseconds.
# lct: The last close time of the lookback candlesticks in milliseconds.
# f: The feature's value
def FEATURES_TABLE_SQL(table_name: IDatabaseTableName) -> str:
    """Returns the Regression Features Table's SQL based on the provided table name.

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
                f   REAL NOT NULL\
            );\
            CREATE INDEX IF NOT EXISTS {table_name}_id ON {table_name}(id);\
            CREATE INDEX IF NOT EXISTS {table_name}_fot ON {table_name}(fot);\
            CREATE INDEX IF NOT EXISTS {table_name}_lct ON {table_name}(lct);\
        "






# Predictions Table
# The table in which the Models' Predictions are stored.
# Column Descriptions:
# id: The identifier of the RegressionModel. F.e: KR_SOME_REGRESSION_ID
# fot: The first open time of the lookback candlesticks in milliseconds.
# lct: The last close time of the lookback candlesticks in milliseconds.
# p: The prediction's dictionary
def PREDICTIONS_TABLE_SQL(table_name: IDatabaseTableName) -> str:
    """Returns the Predictions Table's SQL based on the provided table name.

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
                p   JSONB NOT NULL\
            );\
            CREATE INDEX IF NOT EXISTS {table_name}_id ON {table_name}(id);\
            CREATE INDEX IF NOT EXISTS {table_name}_fot ON {table_name}(fot);\
            CREATE INDEX IF NOT EXISTS {table_name}_lct ON {table_name}(lct);\
        "








# Technical Analysis Table
# The table in which the Technical Analysis data is stored.
# Column Descriptions:
# id: The identifier of the range that is being covered. F.e: FIRSTOT_LASTCT
# ta: The technical analysis dictionary
def TECHNICAL_ANALYSIS_TABLE_SQL(table_name: IDatabaseTableName) -> str:
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
        "name": "features",
        "sql": FEATURES_TABLE_SQL
    },
    {
        "name": "predictions",
        "sql": PREDICTIONS_TABLE_SQL
    },
    {
        "name": "technical_analysis",
        "sql": TECHNICAL_ANALYSIS_TABLE_SQL
    }
]


