from typing import List, Union
from sqlitedict import SqliteDict
from tqdm import tqdm
from modules.database import Database
from modules.model import IPrediction
from modules.prediction_cache import get_arima_pred

print("DATABASE MIGRATION RUNNING:\n")


# Prediction Minification
def _minify_pred(pred: IPrediction) -> IPrediction:
    return { "r": pred["r"],"t": pred["t"],"md": [{"d": pred["md"][0]["d"]}] }



# Init default arima model values
predictions: int = 10
long: float = 0.05
short: float = 0.05


# Initialize the Local DB
db: SqliteDict = SqliteDict("db/db.sqlite", tablename="arima_predictions", outer_stack=False)


# Init the progress bar
progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(db)*2)

# Initialize a Connection and the Cursor
conn = Database.POOL.getconn()
cur = conn.cursor()

# Iterate over each prediction and migrate them to postgres
progress_bar.set_description("Migrating Predictions")
for key, value in db.items():
    # Split the key into id, first_ot and last_ct
    key_split: List[str] = key.split("_")

    # Make sure it is a valid prediction
    if len(key_split) == 3:
        # Save the prediction in postgres
        cur.execute(
            f"\
                INSERT INTO {Database.tn('arima_predictions')}(id, fot, lct, pn, l, s, p) \
                VALUES (%s, %s, %s, %s, %s, %s, %s)\
            ",
            (key_split[0], int(key_split[1]), key_split[2], predictions, long, short, _minify_pred(value))
        )

    # Update the progress bar
    progress_bar.update()

# Commit the write action and put the connection back in the pool
conn.commit()
cur.close()
Database.POOL.putconn(conn)

# Now that all predictions have been migrated, validate their integrities
progress_bar.set_description("Validating Migration Integrity")
for key, value in db.items():
    # Split the key into id, first_ot and last_ct
    key_split: List[str] = key.split("_")

    # Make sure it is a valid prediction
    if len(key_split) == 3:
        # Retrieve the prediction
        pred: Union[IPrediction, None] = get_arima_pred(
            model_id=key_split[0],
            first_ot=int(key_split[1]),
            last_ct=int(key_split[2]),
            predictions=predictions,
            interpreter_long=long,
            interpreter_short=short,
        )

        # Make sure it is an exact match, Otherwise, crash the execution and wipe postgres
        if pred is None or pred != _minify_pred(value):
            print(key)
            print(value)
            print(pred)
            Database.delete_tables()
            raise RuntimeError("The migration was aborted due to a prediction discrepancy.")

    # Update the progress bar
    progress_bar.update()

print("\nDATABASE MIGRATION COMPLETED")