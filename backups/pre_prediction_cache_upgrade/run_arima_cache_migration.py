from typing import List
import os
from tqdm import tqdm
from modules.types import IRegressionPredictionRecord, IPrediction
from modules.database.Database import Database, CURSOR, CONNECTION
from modules.prediction_cache.RegressionPredictionCache import RegressionPredictionCache




# Welcome
os.system('cls' if os.name == 'nt' else 'clear')
print("ARIMA CACHE MIGRATION\n")

# Retrieve the predictions
preds: List[IRegressionPredictionRecord] = Database.read_query("SELECT * FROM arima_predictions")
preds_num: int = len(preds)
if preds_num < 6000000:
    raise RuntimeError(f"There should be at least 6m arima predictions. Only {preds_num} were downloaded")

# Init the migration progress bar
print("1/2) Migrating...\n")
progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=preds_num)

# Iterate over the predictions and insert them into the new table
for pred in preds:
    # Insert the records in the new table
    CURSOR.execute(
        f"\
            INSERT INTO regression_predictions(id, fot, lct, pn, l, s, p) \
            VALUES (%s, %s, %s, %s, %s, %s, %s)\
        ",
        (pred["id"], pred["fot"], pred["lct"], pred["pn"], pred["l"], pred["s"], pred["p"])
    )

    # Update the progress bar
    progress_bar.update()

# Commit the migration
CONNECTION.commit()


# Init the validation progress bar
print("\n2/2) Validating Integrity...\n")
progress_bar_val = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=preds_num)

# Iterate over the predictions and make sure they are identical
for pred_record in preds:
    # Initialize the cache instance and download the prediction
    cache: RegressionPredictionCache = RegressionPredictionCache(
        model_id=pred_record["id"], 
        predictions=pred_record["pn"], 
        interpreter_long=pred_record["l"], 
        interpreter_short=pred_record["s"]
    )
    pred: IPrediction = cache.get(pred_record["fot"], pred_record["lct"])

    # If they are not identical, stop the execution
    if pred_record["p"] != pred:
        print("\nPrediction Record: ")
        print(pred_record)
        print("\nPrediction: ")
        print(pred)
        raise RuntimeError("Prediction Integrity Validation Failed.")

    # Update the progress bar
    progress_bar_val.update()


print("\nARIMA CACHE MIGRATION COMPLETED")