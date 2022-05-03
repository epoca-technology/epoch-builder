from os import makedirs, listdir, remove
from os.path import exists, isfile, join
from typing import List
from sqlitedict import SqliteDict

# Heading
print("DB MERGE\n")

# Init constants
DB_PATH: str = "db_merge"
RESULT_PATH: str = f"{DB_PATH}/result.sqlite"

# Make sure the db path exists
if not exists(DB_PATH):
    makedirs(DB_PATH)
    raise RuntimeError(f"The {DB_PATH} directory must be in the root of the project.")

# If there is a result file, remove it
if isfile(RESULT_PATH):
    remove(RESULT_PATH)

# Make sure there are at least 2 database files in the directory
file_names: List[str] = list(filter(lambda f: ".sqlite" in f, [f for f in listdir(DB_PATH) if isfile(join(DB_PATH, f))]))
if len(file_names) < 2:
    raise RuntimeError("There must be at least 2 database files in order to perform a merge.")

# Initialize the result DB
result_db: SqliteDict = SqliteDict(f"{DB_PATH}/result.sqlite", tablename="arima_predictions", autocommit=False, outer_stack=False)

# Initialize the dbs that will be merged
dbs: List[SqliteDict] = [SqliteDict(f"{DB_PATH}/{fn}", tablename="arima_predictions", outer_stack=False) for fn in file_names]

# Iterate over all dbs and dump the values into the result db
for db in dbs:
    for key, value in db.items():
        result_db[key] = value

# Commit the merge
result_db.commit()

# Print the summary
print(f"Merged ({len(dbs)}):")
for db in dbs:
    print(f"{db.filename}: {len(db)} rows")
print(f"\nResult:")
print(f"{result_db.filename}: {len(result_db)} rows")