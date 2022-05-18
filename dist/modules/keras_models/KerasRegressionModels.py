from typing import Union, Any
from tensorflow import zeros, stack, transpose
from keras import Model, Sequential
from keras.layers import Dense, Flatten, Lambda, Conv1D, MaxPooling1D, LSTM, Dropout, LSTMCell, RNN
from modules.keras_models import IKerasModelConfig, validate





## TRADITIONAL REGRESSION MODELS ##






## Dense Neural Network ##


# DNN_STACK1_LI 
# 1 units:       Dense_1
# 1 activations: Dense_1
def DNN_STACK1_LI(m: IKerasModelConfig) -> Sequential:
    validate(m, 'regression', 'DNN_STACK1_LI', units=1, activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(m["units"][0], activation=m["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(m['predictions'], kernel_initializer=zeros, name="Dense_2"),
        Flatten(name="Flatten_1")
    ])


# DNN_STACK1_FI 
# 1 units:       Dense_1
# 1 activations: Dense_1
def DNN_STACK1_FI(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DNN_STACK1_FI', units=1, activations=1)
    return Sequential([
        Flatten(name="Flatten_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_2"),
        Flatten(name="Flatten_2")
    ])





# DENSE_SOFT_STACK (3 units, 3 activations)
def DENSE_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_SOFT_STACK', units=3, activations=3)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config["units"][1], activation=config["activations"][1], kernel_initializer="normal", name="Dense_2"),
        Dense(config["units"][2], activation=config["activations"][2], kernel_initializer="normal", name="Dense_3"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_4"),
        Flatten(name="Flatten_1")
    ])



# DENSE_MEDIUM_STACK (4 units, 4 activations)
def DENSE_MEDIUM_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_MEDIUM_STACK', units=4, activations=4)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config["units"][1], activation=config["activations"][1], kernel_initializer="normal", name="Dense_2"),
        Dense(config["units"][2], activation=config["activations"][2], kernel_initializer="normal", name="Dense_3"),
        Dense(config["units"][3], activation=config["activations"][3], kernel_initializer="normal", name="Dense_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_5"),
        Flatten(name="Flatten_1")
    ])




# DENSE_HARD_STACK (5 units, 5 activations)
def DENSE_HARD_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_HARD_STACK', units=5, activations=5)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config["units"][1], activation=config["activations"][1], kernel_initializer="normal", name="Dense_2"),
        Dense(config["units"][2], activation=config["activations"][2], kernel_initializer="normal", name="Dense_3"),
        Dense(config["units"][3], activation=config["activations"][3], kernel_initializer="normal", name="Dense_4"),
        Dense(config["units"][4], activation=config["activations"][4], kernel_initializer="normal", name="Dense_5"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_6"),
        Flatten(name="Flatten_1")
    ])








## Convolutional Neural Network ##



# CNN_STACKLESS (1 filters, 1 activations)
def CNN_STACKLESS(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_STACKLESS', filters=1, activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"],), name="Conv1D_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])




# CNN_STACKLESS_MAX_POOLING (1 filters, 1 activations, 1 pool sizes)
def CNN_STACKLESS_MAX_POOLING(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_STACKLESS_MAX_POOLING', filters=1, activations=1, pool_sizes=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"],), name="Conv1D_1"),
        MaxPooling1D(config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])



# CNN_SOFT_STACK (2 filters, 2 activations)
def CNN_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_SOFT_STACK', filters=2, activations=2)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"],), name="Conv1D_1"),
        Conv1D(config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(int(config["lookback"]/2),), name="Conv1D_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])



# CNN_SOFT_STACK_MAX_POOLING (2 filters, 2 activations, 2 pool sizes)
def CNN_SOFT_STACK_MAX_POOLING(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_SOFT_STACK_MAX_POOLING', filters=2, activations=2, pool_sizes=2)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        MaxPooling1D(config["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])


# CNN_SOFT_STACK_MAX_POOLING_DROPOUT (2 filters, 2 activations, 2 pool sizes, 1 dropout rates)
def CNN_SOFT_STACK_MAX_POOLING_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_SOFT_STACK_MAX_POOLING_DROPOUT', filters=2, activations=2, pool_sizes=2, dropout_rates=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        MaxPooling1D(config["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])










## Long Short-Term Memory Recurrent Neural Network ##


# LSTM_STACKLESS (1 units)
def LSTM_STACKLESS(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_STACKLESS', units=1)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_SOFT_STACK (2 units, 1 dropout rates)
def LSTM_SOFT_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_SOFT_STACK_BALANCED_DROPOUT', units=1, dropout_rates=1)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_SOFT_STACK (2 units)
def LSTM_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_SOFT_STACK', units=2)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_MEDIUM_STACK_BALANCED_DROPOUT (4 units, 3 dropout rates)
def LSTM_MEDIUM_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_MEDIUM_STACK_BALANCED_DROPOUT', units=4, dropout_rates=3)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][1], name="Dropout_2"),
        LSTM(config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][2], name="Dropout_3"),
        LSTM(config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_MEDIUM_STACK (4 units, 1 dropout rates)
def LSTM_MEDIUM_STACK(m: IKerasModelConfig) -> Sequential:
    validate(m, 'regression', 'LSTM_MEDIUM_STACK', units=4, activations=4, dropout_rates=1)
    return Sequential([
        LSTM(m["units"][0], activation=m["activations"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], activation=m["activations"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][2], activation=m["activations"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], activation=m["activations"][3], return_sequences=False, name="LSTM_4"),
        Dense(m['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_HARD_STACK_BALANCED_DROPOUT (6 units, 5 dropout rates)
def LSTM_HARD_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_HARD_STACK_BALANCED_DROPOUT', units=6, dropout_rates=5)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][1], name="Dropout_2"),
        LSTM(config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][2], name="Dropout_3"),
        LSTM(config["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(config["dropout_rates"][3], name="Dropout_4"),
        LSTM(config["units"][4], return_sequences=True, name="LSTM_5"),
        Dropout(config["dropout_rates"][4], name="Dropout_5"),
        LSTM(config["units"][5], return_sequences=False, name="LSTM_6"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_HARD_STACK (6 units, 1 dropout rates)
def LSTM_HARD_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_HARD_STACK', units=6, dropout_rates=1)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(config["units"][4], return_sequences=True, name="LSTM_5"),
        LSTM(config["units"][5], return_sequences=False, name="LSTM_6"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])












## AUTOREGRESSIVE REGRESSION MODELS ##







## Dense Neural Network ##




## Convolutional Neural Network ##





## Long Short-Term Memory Recurrent Neural Network ##


class AutoregressiveRegression(Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        #self.lambda_1 = Lambda(lambda x: x[:, -1:, :], name="Lambda_1")
        self.lstm_cell = LSTMCell(units, name="LSTMCELL_1")
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True, name="LSTM_RNN_1")
        self.dense = Dense(4, name="Dense_1")


    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state


    def call(self, inputs, training=None):
        # Flatten the inputs
        #inputs = self._prepare_input(inputs)

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
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = transpose(predictions, [1, 0, 2])
        return predictions





def AR_MOCK_1(m: IKerasModelConfig) -> AutoregressiveRegression:
    return AutoregressiveRegression(50, m["predictions"])