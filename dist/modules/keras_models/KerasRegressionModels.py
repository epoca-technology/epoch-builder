from typing import Union
from tensorflow import zeros
from keras import Model, Sequential
from keras.layers import Dense, Flatten, Lambda, Conv1D, MaxPooling1D, LSTM, Dropout, LSTMCell, RNN
from modules.keras_models import IKerasModelConfig, validate







## TRADITIONAL REGRESSION MODELS ##



## Dense Neural Network ##


# DENSE_STACKLESS (1 units, 1 activations)
def DENSE_STACKLESS(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_STACKLESS', required_units=1, required_activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(units=config["units"][0], activation=config["activations"][0], name="Dense_1"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_2"),
        Flatten(name="Flatten_1")
    ])


# DENSE_SOFT_STACK (3 units, 3 activations)
def DENSE_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_SOFT_STACK', required_units=3, required_activations=3)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(units=config["units"][0], activation=config["activations"][0], name="Dense_1"),
        Dense(units=config["units"][1], activation=config["activations"][1], name="Dense_2"),
        Dense(units=config["units"][2], activation=config["activations"][2], name="Dense_3"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_4"),
        Flatten(name="Flatten_1")
    ])



# DENSE_MEDIUM_STACK (4 units, 4 activations)
def DENSE_MEDIUM_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_MEDIUM_STACK', required_units=4, required_activations=4)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(units=config["units"][0], activation=config["activations"][0], name="Dense_1"),
        Dense(units=config["units"][1], activation=config["activations"][1], name="Dense_2"),
        Dense(units=config["units"][2], activation=config["activations"][2], name="Dense_3"),
        Dense(units=config["units"][3], activation=config["activations"][3], name="Dense_4"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_5"),
        Flatten(name="Flatten_1")
    ])




# DENSE_HARD_STACK (5 units, 5 activations)
def DENSE_HARD_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'DENSE_HARD_STACK', required_units=5, required_activations=5)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(units=config["units"][0], activation=config["activations"][0], name="Dense_1"),
        Dense(units=config["units"][1], activation=config["activations"][1], name="Dense_2"),
        Dense(units=config["units"][2], activation=config["activations"][2], name="Dense_3"),
        Dense(units=config["units"][3], activation=config["activations"][3], name="Dense_4"),
        Dense(units=config["units"][4], activation=config["activations"][4], name="Dense_5"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_6"),
        Flatten(name="Flatten_1")
    ])








## Convolutional Neural Network ##



# CNN_STACKLESS (1 filters, 1 activations)
def CNN_STACKLESS(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_STACKLESS', required_filters=1, required_activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(filters=config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])




# CNN_STACKLESS_MAX_POOLING (1 filters, 1 activations, 1 pool sizes)
def CNN_STACKLESS_MAX_POOLING(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_STACKLESS_MAX_POOLING', required_filters=1, required_activations=1, required_pool_sizes=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(filters=config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(pool_size=config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])



# CNN_SOFT_STACK (2 filters, 2 activations)
def CNN_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_SOFT_STACK', required_filters=2, required_activations=2)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(filters=config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        Conv1D(filters=config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])



# CNN_SOFT_STACK_MAX_POOLING (2 filters, 2 activations, 2 pool sizes)
def CNN_SOFT_STACK_MAX_POOLING(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_SOFT_STACK_MAX_POOLING', required_filters=2, required_activations=2, required_pool_sizes=2)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(filters=config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(pool_size=config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(filters=config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        MaxPooling1D(pool_size=config["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])


# CNN_SOFT_STACK_MAX_POOLING_DROPOUT (2 filters, 2 activations, 2 pool sizes, 1 dropout rates)
def CNN_SOFT_STACK_MAX_POOLING_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'CNN_SOFT_STACK_MAX_POOLING_DROPOUT', required_filters=2, required_activations=2, required_pool_sizes=2, required_dropout_rates=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(filters=config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(pool_size=config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(filters=config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        MaxPooling1D(pool_size=config["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dropout(rate=config["dropout_rates"][0], name="Dropout_1"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])










## Long Short-Term Memory Recurrent Neural Network ##


# LSTM_STACKLESS (1 units)
def LSTM_STACKLESS(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_STACKLESS', required_units=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_SOFT_STACK (2 units, 1 dropout rates)
def LSTM_SOFT_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_SOFT_STACK_BALANCED_DROPOUT', required_units=1, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_SOFT_STACK (2 units)
def LSTM_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_SOFT_STACK', required_units=2)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_MEDIUM_STACK_BALANCED_DROPOUT (4 units, 3 dropout rates)
def LSTM_MEDIUM_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_MEDIUM_STACK_BALANCED_DROPOUT', required_units=4, required_dropout_rates=3)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][1], name="Dropout_2"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][2], name="Dropout_3"),
        LSTM(units=config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_MEDIUM_STACK (4 units, 1 dropout rates)
def LSTM_MEDIUM_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_MEDIUM_STACK', required_units=4, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(units=config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_HARD_STACK_BALANCED_DROPOUT (6 units, 5 dropout rates)
def LSTM_HARD_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_HARD_STACK_BALANCED_DROPOUT', required_units=6, required_dropout_rates=5)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][1], name="Dropout_2"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][2], name="Dropout_3"),
        LSTM(units=config["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(config["dropout_rates"][3], name="Dropout_4"),
        LSTM(units=config["units"][4], return_sequences=True, name="LSTM_5"),
        Dropout(config["dropout_rates"][4], name="Dropout_5"),
        LSTM(units=config["units"][5], return_sequences=False, name="LSTM_6"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_HARD_STACK (6 units, 1 dropout rates)
def LSTM_HARD_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'regression', 'LSTM_HARD_STACK', required_units=6, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(units=config["units"][4], return_sequences=True, name="LSTM_5"),
        LSTM(units=config["units"][5], return_sequences=False, name="LSTM_6"),
        Dense(units=config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])












## AUTOREGRESSIVE REGRESSION MODELS ##




## Dense Neural Network ##




## Convolutional Neural Network ##





## Long Short-Term Memory Recurrent Neural Network ##