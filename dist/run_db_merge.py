from os import makedirs, listdir, remove
from os.path import exists, isfile, join
from typing import List
from sqlitedict import SqliteDict
from modules.model import IPrediction

# Heading
print("DB MERGE\n")

print("Merging...\n")

# Init constants
DB_MERGE_PATH: str = "db_merge"
RESULT_PATH: str = f"db/merge_result.sqlite"
LOCAL_PATH: str = f"db/db.sqlite"

# Make sure the db path exists, otherwise create it, raise and error and provide instrutions
if not exists(DB_MERGE_PATH):
    makedirs(DB_MERGE_PATH)
    raise RuntimeError(f"The {DB_MERGE_PATH} directory must be in the root of the project and contain at least 1 db file.")

# If there is a result file, raise an error
if isfile(RESULT_PATH):
    raise RuntimeError(f"The file {RESULT_PATH} cannot exist prior to running a DB Merge.")

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
for db in dbs + [local_db]:
    for key, value in db.items():
        result_db[key] = value

# Commit the merge
result_db.commit()

# Calculate the # of items in the local and the merge result dbs
local_db_len: int = len(local_db)
result_db_len: int = len(result_db)
db_lens: List[int] = [local_db_len]

# Print the summary and close the connections
print(f"Merged ({len(dbs) + 1}):")
print(f"{local_db.filename}: {local_db_len} rows")
for db in dbs:
    db_len: int = len(db)
    db_lens.append(db_len)
    print(f"{db.filename}: {db_len} rows")
    db.close()
print(f"\nResult:")
print(f"{result_db.filename}: {len(result_db)} rows")

# Close the local db
local_db.close()

# Close the result db
result_db.close()



## Integrity Validation ##
print("\nValidating Integrity...\n")

# Init Helpers

def _is_prediction(pred: IPrediction) -> bool:
    """Checks if a stored prediction is valid.

    Args:
        pred: IPrediction
    
    Returns: 
        bool
    """
    return isinstance(pred, dict) \
        and isinstance(pred.get('r'), int) \
            and isinstance(pred.get('t'), int) \
                and isinstance(pred.get('md'), list) \
                    and len(pred['md']) >= 1 \
                        and isinstance(pred['md'][0], dict)



# The merge result must have the highest number of rows
if result_db_len < max(db_lens):
    raise RuntimeError(f"The result db does not have the highest number of rows. {result_db_len} < {max(db_lens)}")

# Reopen the merge result db
result_db = SqliteDict(RESULT_PATH, tablename="arima_predictions", outer_stack=False)

# The len must be identical to the one recorded before closing the connection
if result_db_len != len(result_db):
    raise RuntimeError(f"The re-opened result db does not have the same amount of rows as it had before closing the \
        connection. {result_db_len} != {len(result_db)}")

# Iterate over each item and make sure the content is valid
for key, item in result_db.items():
    if not _is_prediction(item):
        raise RuntimeError(f"The prediction {key} contents do not follow the correct structure.")





## Clean Up ##


# Clean the db_merge directory
for fn in file_names:
    remove(f"{DB_MERGE_PATH}/{fn}")




print("DB MERGE COMPLETED")