from typing import Any, Optional, Tuple, List, Union
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor, Json
from psycopg2.extensions import new_type, DECIMAL, register_type, register_adapter
from modules.database.DatabaseTables import TABLES
from modules.model import IPrediction







## DATABASE CONFIGURATION ##



## Cursors ## 



# Dict Cursor - Used to retrieve data in a dictionary format that can be accessed
# by column name
DICT_CURSOR: RealDictCursor = RealDictCursor






## Data Adapters ##


# DECIMAL
# Psycopg converts decimal / numeric database types into Python Decimal objects. 
# This adapter convers these values into floats.
DEC2FLOAT = new_type(
    DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: float(value) if value is not None else None)
register_type(DEC2FLOAT)



# JSON
# Registers the Json Adapter
register_adapter(dict, Json)








## DATABASE MANAGER ##


class Database:
    """Database Class

    This singleton manages the interactions with the PostgreSQL Database.

    Class Properties:
        TEST_MODE: bool
            The type of execution.
        POOL: SimpleConnectionPool
            Pool of connections.
    """

    # TEST MODE
    # If TEST_MODE is enabled, the db module will use the test table names instead of the 
    # real ones in order to prevent potential incidents.
    TEST_MODE: bool = False



    # Pool Connection
    # A connection can be requested from the pool at any time.
    # Make sure to perform the following actions once the actions complete:
    # cursor.close()
    # pool.putconn(connection)
    POOL: SimpleConnectionPool = SimpleConnectionPool(
        3,
        10,
        host="192.168.2.101",
        user="postgres",
        password="oPyjNQqeP8LewFFELIA2BuwoTi8RkYDLaAvhyvWT",
        database="postgres",
        port="5442"
    )





    ## Database Queries ## 



    @staticmethod
    def read_query(text: str, values: Optional[Tuple[Any]] = None) -> List[Any]:
        """Executes a read query and returns whatever is returned by the DB Driver. Note that the return
        value may be None.
        
        Once the query executes, it closes the cursor and returns the connection to the pool.
        
        Args:
            text (str): The query to be executed.
            values? (Tuple[Any]): The values to be used for the query substitutions.

        Returns:
            List[Any]
        """
        # Initialize a Connection and the Cursor
        conn = Database.POOL.getconn()
        cur = conn.cursor(cursor_factory=DICT_CURSOR)

        # Execute the query
        try:
            # Handle the case accordingly
            if values:
                cur.execute(text, values)
            else:
                cur.execute(text)
            
            # Return the Execution Response
            return cur.fetchall()
        finally:
            cur.close()
            Database.POOL.putconn(conn)








    @staticmethod
    def write_query(text: str, values: Optional[Tuple[Any]] = None) -> None:
        """Executes a write query as well as commiting the changes. Once the query executes, it closes the cursor
        and returns the connection to the pool.

        Args:
            text (str): The query to be executed.
            values? (Tuple[Any]): The values to be used for the query substitutions.
        """
        # Initialize a Connection and the Cursor
        conn = Database.POOL.getconn()
        cur = conn.cursor()

        # Execute the query
        try:
            # Handle the case accordingly
            if values:
                cur.execute(text, values)
            else:
                cur.execute(text)
            
            # Commit the write action
            conn.commit()
        finally:
            cur.close()
            Database.POOL.putconn(conn)








    ## DATABASE TABLES MACRO MANAGEMENT ##



    @staticmethod
    def initialize_tables() -> None:
        """Creates all the required db tables in case they haven't already been.
        """
        # Iterate over each table
        for table in TABLES:
            # Handle the table
            Database.write_query(table["sql"](table["name"]))

            # Handle the test table
            Database.write_query(table["sql"]("test_" + table["name"]))





    @staticmethod
    def delete_tables() -> None:
        """Deletes all the db tables prior to restoration.
        """
        # Build the table names inline
        table_union: str = ""
        for i, table in enumerate(TABLES):
            # Handle the last item accordingly
            if i == len(TABLES) - 1:
                table_union += f"{table['name']}, test_{table['name']};"

            # Otherwise, just append the table and its test
            else:
                table_union += f"{table['name']}, test_{table['name']}, "


        # Delete them from the db
        Database.write_query(f"DROP TABLE IF EXISTS ${table_union}")










    ## TABLE NAME RETRIEVER ##



    @staticmethod
    def tn(table_name: str) -> str:
        """Retrieves a table name based on the current TEST_MODE Value.
        If test_mode is enabled, it will attach the test_ string to the provided name.
        F.e test_$TABLE_NAME

        Args:
            table_name (str): 
                The name of the table.

        Returns:
            str
        """
        return table_name if not Database.TEST_MODE else f"test_{table_name}"











## DATABASE INITIALIZATION ##
# When the Database module is initialized, make sure that all tables exist.
Database.initialize_tables()











## LEGACY DB IMPLEMENTATION ##
# This implementation has been deprecated in favor of a PostgreSQL Database and will be 
# removed once the migration is completed and tested.


import os
from sqlitedict import SqliteDict


# If the Database's directory doesn't exist, create it
DB_PATH: str = 'db'
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)


## Database Init ##
DB: SqliteDict = SqliteDict(f"{DB_PATH}/db.sqlite", tablename="arima_predictions", autocommit=True, outer_stack=False)





# Predictions Management
# In order to accelerate the execution times of the backtesting, predictions are saved in a 
# local database in a key: val format based on the id of the model, the first ot and the last
# ct of the lookback range.
# An example of a key that holds a prediction is: A601_1502942400000_1509139799999
# The keys may be longer in Regression and Classification Models. So far, the limit of the keys
# is unknown. However, they impact performance.




def save_pred(model_id: str, first_ot: int, last_ct: int, pred: IPrediction) -> None:
    """Saves a Model Prediction in the database for optimization purposes.

    Args:
        model_id: str
            The ID of the Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.
        pred: IPrediction
            The prediction to save in the db.
    """
    DB[_get_pred_key(model_id, first_ot, last_ct)] = pred





def get_pred(model_id: str, first_ot: int, last_ct: int) -> Union[IPrediction, None]:
    """Retrieves a Model Prediction if it exists, otherwise returns None.

    Args:
        model_id: str
            The ID of the Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.

    Returns:
        Union[IPrediction, None]
    """
    return DB.get(_get_pred_key(model_id, first_ot, last_ct))





def delete_pred(model_id: str, first_ot: int, last_ct: int) -> None:
    """Deletes a Model Prediction from the Database.

    Args:
        model_id: str
            The ID of the Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.
    """
    # Init the key
    key: str = _get_pred_key(model_id, first_ot, last_ct)

    # if the record exists, delete it
    if key in DB:
        del DB[key]




def _get_pred_key(model_id: str, first_ot: int, last_ct: int) -> str:
    """Returns the key that should be used to save or retrieve the prediction.

    Args:
        model_id: str
            The ID of the Arima Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.
    
    Returns:
        str
    """
    return f"{model_id}_{first_ot}_{last_ct}"