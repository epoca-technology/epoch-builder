from typing import List, Dict
from os.path import exists, isfile
from json import dumps
from tqdm import tqdm
from modules.candlestick import Candlestick


## CLI WELCOME ##
print("LOOKBACK PREDICTION RANGE INDEXER RUNNING\n")

# Make sure the indexer does not exist
if not exists(Candlestick.BASE_PATH):
    raise RuntimeError("The candlesticks directory does not exist.")
if isfile(f"{Candlestick.BASE_PATH}/{Candlestick.INDEXER_NAME}.json"):
    raise RuntimeError("The indexer file already exists in the candlesticks directory.")

# Initialize the list of lookbacks ordered by size
lookbacks: List[int] = [ 100, 300 ]

# Initialize the candlesticks with the max lookback
Candlestick.init(lookbacks[-1])

# Initialize the indexer that will hold the values
indexer: Dict[str, List[int]] = {}

# Init the progress bar
progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=len(lookbacks)*Candlestick.DF.shape[0])

# Iterate over each lookback
for lookback in lookbacks:
    for _, candlestick in Candlestick.DF.iterrows():
        # Init the ID of the range
        id: str = Candlestick._get_lookback_prediction_range_id(lookback, candlestick["ot"])

        # Calculate the range
        first_ot, last_ct = Candlestick._calculate_lookback_prediction_range(lookback, candlestick["ot"])

        indexer[id] = [first_ot, last_ct]

        # Update the progress bar
        progress_bar.update()


# Finally, Save the File
with open(f"{Candlestick.BASE_PATH}/{Candlestick.INDEXER_NAME}.json", "w") as outfile:
    outfile.write(dumps(indexer))
print("\nLOOKBACK PREDICTION RANGE INDEXER COMPLETED")