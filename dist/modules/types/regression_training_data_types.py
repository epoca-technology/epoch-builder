from typing import Tuple
from numpy import ndarray



# Datasets to train and test a Trainable Regression Model
# (train_x, train_y, test_x, test_y)
IRegressionDatasets = Tuple[ndarray, ndarray, ndarray, ndarray]