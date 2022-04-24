from typing import TypedDict, List, Tuple
from pandas import DataFrame
from random import random


# Row Interface
class IRow(TypedDict):
    # Arima Features
    # The columns values can be the following:
    # 0: Neutral
    # 1: Short
    # 2: Long
    arima01: int
    arima02: int
    arima03: int
    arima04: int
    arima05: int

    # Labels
    # Vales can be 1 or 0 based on the outcome
    up: int
    down: int



# Helper Functions
def _get_row(index: int) -> IRow:
    if index % 2:
        return _get_random_row()
    elif index % 3:
        return {
            'arima01': 2,
            'arima02': 0,
            'arima03': 2,
            'arima04': 1,
            'arima05': 2,
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
            'up': 0,
            'down': 1,
        }
    else:
        return _get_random_row()


def _get_random_row() -> IRow:
    up, down = _get_random_labels()
    return {
        'arima01': _get_random_feature(),
        'arima02': _get_random_feature(),
        'arima03': _get_random_feature(),
        'arima04': _get_random_feature(),
        'arima05': _get_random_feature(),
        'up': up,
        'down': down,
    }

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
df: DataFrame = DataFrame(data={'arima01': [], 'arima02': [], 'arima03': [], 'arima04': [], 'arima05': [], 'up': [], 'down': []})

# Input the row count
row_count: int = int(input("Enter the number of rows: "))

# Build the random rows
rows: List[IRow] = [_get_row(i) for i in range(row_count)]

# Append them to the DF
df = df.append(rows)

# Convert all the values to integers
df = df.astype(int)

# Create the CSV File
df.to_csv('csv_mock.csv', index=False)