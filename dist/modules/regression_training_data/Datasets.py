from typing import Union, List
from numpy import ndarray, array
from pandas import DataFrame
from modules._types import IRegressionDatasets
from modules.candlestick.Candlestick import Candlestick





def make_datasets(lookback: int, autoregressive: bool, predictions: int, train_split: float) -> IRegressionDatasets:
    """Builds a tuple containing the features and labels for the train and test datasets based
    on the kind of regression. 

    Args:
        lookback: int
            The number of prediction candlesticks the model needs to look at in order
            to make a prediction.
        autoregressive: bool
            The kind of regression. An Autoregressive Regression performs predictions 1 by 
            1 and keeps feeding them to itself. On the other hand, a single shot prediction
            generates them all at once.
            The training data looks as follows:
            Autoregressive: features[100] -> labels[1]
            Single Shot: features[100] -> labels[30]
        predictions: int
            The number of predictions the model will output in the end.
        train_split: float
            The split that should be applied to the training / test datasets.

    Returns:
        IRegressionDatasets
        (train_x, train_y, test_x, test_y)
    """
    # Init the df, grabbing only the close prices
    df: DataFrame = Candlestick.NORMALIZED_PREDICTION_DF[["c"]].copy()

    # Init the number of rows and the split that will be applied
    rows: int = df.shape[0]
    split: int = int(rows * train_split)

    # Init raw features and labels
    features_raw: Union[List[List[float]], ndarray] = []
    labels_raw: Union[List[List[float]], ndarray] = []

    # Iterate over the normalized ds and build the features & labels
    for i in range(lookback, rows):
        # If it is an autoregression, add only 1 price as the label
        if autoregressive:
            features_raw.append(df.iloc[i-lookback:i, 0])
            labels_raw.append(df.iloc[i, 0])

        # If it is not an autoregression, add the labels based on the number of predictions
        elif not autoregressive and i < (rows-predictions):
            features_raw.append(df.iloc[i-lookback:i, 0])
            labels_raw.append(df.iloc[i:i+predictions, 0])

    # Convert the features and labels into np arrays
    features = array(features_raw)
    labels = array(labels_raw)

    # Finally, return the split datasets
    return features[:split], labels[:split], features[split:], labels[split:]