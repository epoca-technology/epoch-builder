from typing import TypedDict, List, Tuple
from pandas import DataFrame
from random import random


# Init features and labels
features: List[str] = ['arima01', 'arima02', 'arima03', 'arima04', 'arima05', 'arima06', 'arima07', 'arima08', 'arima09', 'arima10' ]
labels: List[str] = ['up', 'down']



# Helper Functions
def _get_row(index: int) -> dict:
    if index % 2:
        return _get_random_row()
    elif index % 3:
        return {
            'arima01': 2,
            'arima02': 0,
            'arima03': 2,
            'arima04': 1,
            'arima05': 2,
            'arima06': 2,
            'arima07': 0,
            'arima08': 1,
            'arima09': 2,
            'arima10': 2,
            'up': 1,
            'down': 0,
        }
    elif index % 5:
        return {
            'arima01': 0,
            'arima02': 1,
            'arima03': 1,
            'arima04': 1,
            'arima05': 2,
            'arima06': 0,
            'arima07': 1,
            'arima08': 2,
            'arima09': 1,
            'arima10': 0,
            'up': 0,
            'down': 1,
        }
    else:
        return _get_random_row()


def _get_random_row() -> dict:
    # Init the row
    row = {nm: _get_random_feature() for nm in features}

    # Retrieve the label values and add them to the dict
    up, down = _get_random_labels()
    row['up'] = up
    row['down'] = down

    # Finally, return the row
    return row

def _get_random_feature() -> int:
    random_val: float = random()
    if random_val <= 0.45:
        return 1
    elif random_val > 0.45 and random_val < 0.55:
        return 0
    else:
        return 2

def _get_random_labels() -> Tuple[int, int]:
    if random() > 0.5:
        return 1, 0
    else:
        return 0, 1



# Initialize the DF
df: DataFrame = DataFrame(data={nm: [] for nm in features + labels})
#df: DataFrame = DataFrame(data={'arima01':[],'arima02':[],'arima03':[],'arima04':[],'arima05':[],'arima06': [],'arima07': [],'arima08': [],'arima09': [],'arima10': [],'up': [], 'down': []})

# Input the row count
row_count: int = int(input("Enter the number of rows: "))

# Build the random rows
rows: List[dict] = [_get_row(i) for i in range(row_count)]

# Append them to the DF
df = df.append(rows)

# Convert all the values to integers
df = df.astype(int)

# Create the CSV File
df.to_csv('decision_data/decision_data_dump.csv', index=False)