from typing import Any, Optional, Tuple, List
from psycopg2 import connect
from psycopg2.extras import RealDictCursor, Json
from psycopg2.extensions import new_type, DECIMAL, register_type, register_adapter
from modules._types import IDatabaseConnectionConfig, IDatabaseSummary, IDatabaseTableSummary,\
    IDatabaseTableName, IDatabaseTableNameInput
from modules.database.DatabaseTables import TABLES







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






## DATABASE CONNECTION ##


# CONFIGURATION
# This is the configuration that will be used to establish a connection with the
# database.
DB_CONNECTION_CONFIG: IDatabaseConnectionConfig = {
    "host_ip": "192.168.2.101",
    "user": "postgres",
    "password": "oPyjNQqeP8LewFFELIA2BuwoTi8RkYDLaAvhyvWT",
    "database": "postgres",
    "port": "5442",
}




# CONNECTION
# The established connection with the database. This instance can be used to 
# generate cursors or commit writes.
CONNECTION: Any = connect(
    host=DB_CONNECTION_CONFIG["host_ip"],
    user=DB_CONNECTION_CONFIG["user"],
    password=DB_CONNECTION_CONFIG["password"],
    database=DB_CONNECTION_CONFIG["database"],
    port=DB_CONNECTION_CONFIG["port"]
)




# CONNECTION CURSOR
# A ready to go connection cursor.
CURSOR: Any = CONNECTION.cursor(cursor_factory=DICT_CURSOR)







## DATABASE SINGLETON ##


class Database:
    """Database Class

    This singleton manages the interactions with the PostgreSQL Database.

    Class Properties:
        TEST_MODE: bool
            The type of execution.
        DB_MANAGEMENT_PATH: str
            This is the path where the backup and restore actions should place or read files
            from.
    """

    # Test Mode
    # If TEST_MODE is enabled, the db module will use the test table names instead of the 
    # real ones in order to prevent potential incidents.
    TEST_MODE: bool = False



    # Database Management Path
    # This is the path where the backup and restore actions should place or read files
    # from.
    DB_MANAGEMENT_PATH: str = "db_management"










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
        # Handle the case accordingly
        if values:
            CURSOR.execute(text, values)
        else:
            CURSOR.execute(text)
        
        # Return the Execution Response
        return CURSOR.fetchall()








    @staticmethod
    def write_query(text: str, values: Optional[Tuple[Any]] = None) -> None:
        """Executes a write query as well as commiting the changes. Once the query executes, it closes the cursor
        and returns the connection to the pool.

        Args:
            text (str): The query to be executed.
            values? (Tuple[Any]): The values to be used for the query substitutions.
        """
        # Handle the case accordingly
        if values:
            CURSOR.execute(text, values)
        else:
            CURSOR.execute(text)
        
        # Commit the write action
        CONNECTION.commit()








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
        Database.write_query(f"DROP TABLE IF EXISTS {table_union}")










    ## TABLE NAME RETRIEVER ##



    @staticmethod
    def tn(table_name: IDatabaseTableNameInput) -> IDatabaseTableName:
        """Retrieves a table name based on the current TEST_MODE Value.
        If test_mode is enabled, it will attach the test_ string to the provided name.
        F.e test_$TABLE_NAME

        Args:
            table_name: IDatabaseTableNameInput
                The name of the table.

        Returns:
            IDatabaseTableName
        """
        return table_name if not Database.TEST_MODE else f"test_{table_name}"






    ## DATABASE SUMMARY ##


    @staticmethod
    def get_summary() -> IDatabaseSummary:
        """Retrieves the summary of the database and its configuration.

        Returns:
            IDatabaseSummary
        """
        # Retrieve the version of the database
        version_snap: List[Any] = Database.read_query("SELECT version();")

        # Retrieve the total size
        total_size_snap: List[Any] = Database.read_query(f"SELECT pg_database_size('{DB_CONNECTION_CONFIG['database']}');")

        # Retrieve the tables' summaries
        tables: List[IDatabaseTableSummary] = []
        test_tables: List[IDatabaseTableSummary] = []

        # Iterate over each table and build the summaries
        for table in TABLES:
            # Build the table summary
            table_size_snap: List[Any] = Database.read_query(f"SELECT pg_total_relation_size('{table['name']}');")
            tables.append({
                "name": table["name"],
                "size": table_size_snap[0]["pg_total_relation_size"]
            })

            # Build the test table summary
            test_table_size_snap: List[Any] = Database.read_query(f"SELECT pg_total_relation_size('test_{table['name']}');")
            test_tables.append({
                "name": "test_" + table["name"],
                "size": test_table_size_snap[0]["pg_total_relation_size"]
            })

        # Finally, return the summary
        return {
            "connection_config": DB_CONNECTION_CONFIG,
            "version": version_snap[0]["version"],
            "size": total_size_snap[0]["pg_database_size"],
            "tables": tables,
            "test_tables": test_tables,
        }





## DATABASE INITIALIZATION ##
# When the Database module is initialized, make sure that all tables exist.
Database.initialize_tables()