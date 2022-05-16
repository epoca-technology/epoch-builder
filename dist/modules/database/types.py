from typing import Callable, TypedDict, List






# Database Connection Configuration
# This is the configuration that will be used to establish a connection with the
# database.
class IDatabaseConnectionConfig(TypedDict):
    host_ip: str
    user: str
    password: str
    database: str
    port: str





# Database Table Type
# Includes the base name of the table as well as the sql to create it safely
class IDatabaseTable(TypedDict):
    name: str
    sql: Callable[[str], str]





# Database Summary
# General information about the Database and its configuration.

class IDatabaseTableSummary(TypedDict):
    name: str
    size: float

class IDatabaseSummary(TypedDict):
    connection_config: IDatabaseConnectionConfig
    version: str
    size: float
    tables: List[IDatabaseTableSummary]
    test_tables: List[IDatabaseTableSummary]