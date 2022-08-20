from typing import Tuple, TypedDict
from numpy import ndarray



# Datasets to train and test a Trainable Regression Model
# (train_x, train_y, test_x, test_y)
IRegressionDatasets = Tuple[ndarray, ndarray, ndarray, ndarray]




# Regression Dataset Summary
# This summary is extracted directly from the Dataset
class IRegressionDatasetSummary(TypedDict):
    count: float
    mean: float
    std: float
    min: float
    #"25%": float
    #"50%": float
    #"75%": float
    max: float