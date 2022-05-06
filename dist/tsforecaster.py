from typing import Tuple, Union, TypedDict
from pandas import DataFrame, Series, read_csv, set_option
from numpy import arange, array, float32
from tensorflow import stack, newaxis, zeros, tile, transpose
from keras import Model, Sequential
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.layers import Dense, Flatten, Reshape, Conv1D, LSTM, Lambda, LSTMCell, RNN
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from modules.utils import Utils


# Make Panda's floats more readable
set_option('display.float_format', lambda x: '%.2f' % x)


# Initialize the DataFrame
raw_df: DataFrame = read_csv("candlesticks/prediction_candlesticks.csv", usecols=("o", "h", "l", "c"))



# Split the data
column_indices = {name: i for i, name in enumerate(raw_df.columns)}
n = len(raw_df)
train_df = raw_df[0:int(n*0.7)]
val_df = raw_df[int(n*0.7):int(n*0.9)]
test_df = raw_df[int(n*0.9):]

num_features = raw_df.shape[1]



# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

"""
print(train_df.head())
print(val_df.head())
print(test_df.head())
"""

# Data windowing

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = { name: i for i, name in enumerate(label_columns) }
    self.column_indices = { name: i for i, name in enumerate(train_df.columns) }

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])



def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window




## Example Windows ##

w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['c'])

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['c'])



# Stack three slices, the length of the total window.
example_window = stack([array(train_df[:w2.total_window_size]),
                           array(train_df[100:100+w2.total_window_size]),
                           array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

"""
print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
"""




## Create tf.data.Datasets ##

def make_dataset(self, data):
  data = array(data, dtype=float32)
  ds = timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example



## Single step models ##


single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['c'])



## Baseline ##



class Baseline(Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, newaxis]


baseline = Baseline(label_index=column_indices['c'])

baseline.compile(loss=MeanSquaredError(),
                 metrics=[MeanAbsoluteError()])
"""
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
"""





## Wide Window ## 

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['c'])





## Model Training ##
MAX_EPOCHS = 20
def compile_and_fit(model, window, patience=2):
  early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')

  model.compile(loss=MeanSquaredError(),
                optimizer=adam_v2.Adam(),
                metrics=[MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history




## Linear model ##

linear = Sequential([
    Dense(units=1)
])


"""
history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
"""





## Dense ##

dense = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=1)
])

"""
history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
"""






## Multi-step dense ##

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['c'])


multi_step_dense = Sequential([
    # Shape: (time, features) => (time*features)
    Flatten(),
    Dense(units=32, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    Reshape([1, -1]),
])


"""
history = compile_and_fit(multi_step_dense, conv_window)


val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
"""






## Convolution neural network ##



conv_model = Sequential([
        Conv1D(filters=32,kernel_size=(CONV_WIDTH,),activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1),
])

"""
history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
"""




## Wide Convolution neural network ##


LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['c'])

#print("Wide conv window")
#print('Input shape:', wide_conv_window.example[0].shape)
#print('Labels shape:', wide_conv_window.example[1].shape)
#print('Output shape:', conv_model(wide_conv_window.example[0]).shape)







## Recurrent neural network ##


lstm_model = Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    Dense(units=1)
])

"""
history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

"""



## Multi-output models ##




single_step_window = WindowGenerator(
    # `WindowGenerator` returns all features as labels if you 
    # don't set the `label_columns` argument.
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)



baseline = Baseline()
baseline.compile(loss=MeanSquaredError(),
                 metrics=[MeanAbsoluteError()])


val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(wide_window.val)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)






## Dense ## 


dense = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_features)
])


"""
history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

"""


## RNN ##

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)

lstm_model = Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    Dense(units=num_features)
])


"""
history = compile_and_fit(lstm_model, wide_window)


val_performance['LSTM'] = lstm_model.evaluate( wide_window.val)
performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0)
"""






## Multi-step models ##



OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS)




class MultiStepLastBaseline(Model):
  def call(self, inputs):
    return tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=MeanSquaredError(),
                      metrics=[MeanAbsoluteError()])


multi_val_performance = {}
multi_performance = {}

"""
multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
"""


class RepeatBaseline(Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()


repeat_baseline.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

"""
multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
"""



## Linear ##


multi_linear_model = Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    Dense(OUT_STEPS*num_features, kernel_initializer=zeros),
    # Shape => [batch, out_steps, features]
    Reshape([OUT_STEPS, num_features])
])


"""
history = compile_and_fit(multi_linear_model, multi_window)

multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
"""



## Dense ##

multi_dense_model = Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    Dense(OUT_STEPS*num_features, kernel_initializer=zeros),
    # Shape => [batch, out_steps, features]
    Reshape([OUT_STEPS, num_features])
])

"""
history = compile_and_fit(multi_dense_model, multi_window)


multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
"""



## CNN ##


CONV_WIDTH = 3
multi_conv_model = Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    Dense(OUT_STEPS*num_features, kernel_initializer=zeros),
    # Shape => [batch, out_steps, features]
    Reshape([OUT_STEPS, num_features])
])


"""
history = compile_and_fit(multi_conv_model, multi_window)


multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
"""



## RNN ##


multi_lstm_model = Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    Dense(OUT_STEPS*num_features, kernel_initializer=zeros),
    # Shape => [batch, out_steps, features].
    Reshape([OUT_STEPS, num_features])
])

"""
history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
"""



## Autoregressive model ##



class FeedBack(Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
    self.dense = Dense(num_features)


feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup



def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the LSTM state.
  prediction, state = self.warmup(inputs)

  # Insert the first prediction.
  predictions.append(prediction)

  # Run the rest of the prediction steps.
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output.
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call




history = compile_and_fit(feedback_model, multi_window)

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)