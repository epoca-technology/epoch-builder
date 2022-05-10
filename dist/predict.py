from typing import Tuple, Union, TypedDict
from pandas import DataFrame, Series, read_csv, set_option
from numpy import arange, array, float32, ndarray
from tensorflow import stack, newaxis, zeros, tile, transpose, data as tfdata, Tensor
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Model, Sequential
from keras.models import load_model
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.layers import Dense, Flatten, Reshape, Conv1D, LSTM, Lambda, LSTMCell, RNN
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from h5py import File
from modules.candlestick import Candlestick
from modules.utils import Utils


# Make Panda's floats more readable
set_option('display.float_format', lambda x: '%.6f' % x)




# Load the Model
#model: Sequential = load_model('./regression_models/LSTM1_REG_300_10.h5')
#print(model.summary())
id: str = ''
lookback: int = 0
predictions: int = 0
with File('./regression_models/LSTM_SOFT_STACK_BALANCED_DROPOUT_LB50_P5_RMSP_LR001_MSE/model.h5', mode='r') as f:
    id = f.attrs['id']
    lookback = f.attrs['lookback']
    predictions = f.attrs['predictions']
    model = load_model_from_hdf5(f)

print(id)
print(lookback)
print(predictions)


# Initialize the candlesticks
Candlestick.init(lookback, normalized_df=True)


# Init the lookback df
df: DataFrame = Candlestick.get_lookback_df(lookback, Candlestick.DF.iloc[875789]['ot'], normalized=True)





def make_dataset(data: DataFrame, lookback: int) -> tfdata.Dataset:
    """Converts a DataFrame into a Dataset.

    Args:
        data: DataFrame
            The data to be converted into a TF Dataset
    
    Returns:
        tfdata.Dataset
    """
    # Convert the DataFrame into a numpy array
    data: ndarray = array(data, dtype=float32)

    # Initialize the Dataset
    ds: tfdata.Dataset = timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=lookback,
        sequence_stride=1,
        shuffle=False,
        batch_size=1
    )

    # Split the Dataset into windows
    return ds


print("\n", df.tail(20))
preds = model.predict(make_dataset(df, lookback))[0]
print("\n", preds)
print("\n", Utils.get_percentage_change(preds[0], preds[-1]))

