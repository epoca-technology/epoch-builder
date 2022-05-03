from os import makedirs, listdir, remove
from os.path import exists, isfile, join
from typing import List
from sqlitedict import SqliteDict

# Heading
print("DB MERGE\n")

# Init constants
DB_MERGE_PATH: str = "db_merge"
RESULT_PATH: str = f"db/merge_result.sqlite"
LOCAL_PATH: str = f"db/db.sqlite"

# Make sure the db path exists
if not exists(DB_MERGE_PATH):
    makedirs(DB_MERGE_PATH)
    raise RuntimeError(f"The {DB_MERGE_PATH} directory must be in the root of the project and contain at least 1 db file.")

# If there is a result file, remove it
if isfile(RESULT_PATH):
    remove(RESULT_PATH)

# Extract the names of the databases that will be merged
file_names: List[str] = list(filter(lambda f: ".sqlite" in f, [f for f in listdir(DB_MERGE_PATH) if isfile(join(DB_MERGE_PATH, f))]))
if len(file_names) < 1:
    raise RuntimeError("There must be at least 1 database file in order to perform a merge.")

# Initialize the result DB
result_db: SqliteDict = SqliteDict(RESULT_PATH, tablename="arima_predictions", autocommit=False, outer_stack=False)

# Initialize the Local DB
local_db: SqliteDict = SqliteDict(LOCAL_PATH, tablename="arima_predictions", outer_stack=False)

# Initialize the dbs that will be merged
dbs: List[SqliteDict] = [SqliteDict(f"{DB_MERGE_PATH}/{fn}", tablename="arima_predictions", outer_stack=False) for fn in file_names]

# Iterate over all dbs and dump the values into the result db
for db in dbs:
    for key, value in db.items():
        result_db[key] = value

# Commit the merge
result_db.commit()

# Print the summary
print(f"Merged ({len(dbs) + 1}):")
print(f"{local_db.filename}: {len(local_db)} rows")
for db in dbs:
    print(f"{db.filename}: {len(db)} rows")
print(f"\nResult:")
print(f"{result_db.filename}: {len(result_db)} rows")


# Clean the db_merge directory
for fn in file_names:
    remove(f"{DB_MERGE_PATH}/{fn}")