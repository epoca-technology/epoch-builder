from typing import Callable, Literal, TypedDict, List






# Database Connection Configuration
# This is the configuration that will be used to establish a connection with the
# database.
class IDatabaseConnectionConfig(TypedDict):
    host_ip: str
    user: str
    password: str
    database: str
    port: str







# Database Table Names Input & Output
IDatabaseTableNameInput = Literal["classification_predictions", "regression_predictions", "technical_analysis"]
IDatabaseTableName = Literal[
    "classification_predictions", "regression_predictions", "technical_analysis",
    "test_classification_predictions", "test_regression_predictions", "test_technical_analysis"
]




# Database Table Type
# Includes the base name of the table as well as the sql to create it safely
class IDatabaseTable(TypedDict):
    name: IDatabaseTableNameInput
    sql: Callable[[str], str]






# Database Summary
# General information about the Database and its configuration.

class IDatabaseTableSummary(TypedDict):
    name: IDatabaseTableName
    size: float

class IDatabaseSummary(TypedDict):
    connection_config: IDatabaseConnectionConfig
    version: str
    size: float
    tables: List[IDatabaseTableSummary]
    test_tables: List[IDatabaseTableSummary]