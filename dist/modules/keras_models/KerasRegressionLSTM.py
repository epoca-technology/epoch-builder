from typing import Union
from tensorflow import zeros
from keras import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, LSTMCell, RNN
from modules.keras_models import IKerasModelConfig, validate







## Standard Regression Models ##



# LSTM_STACKLESS (1 units)
def LSTM_STACKLESS(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_STACKLESS', required_units=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_SOFT_STACK (2 units, 1 dropout rates)
def LSTM_SOFT_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_SOFT_STACK_BALANCED_DROPOUT', required_units=1, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_SOFT_STACK (2 units)
def LSTM_SOFT_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_SOFT_STACK', required_units=2)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_MEDIUM_STACK_BALANCED_DROPOUT (4 units, 3 dropout rates)
def LSTM_MEDIUM_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_MEDIUM_STACK_BALANCED_DROPOUT', required_units=4, required_dropout_rates=3)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][1], name="Dropout_2"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][2], name="Dropout_3"),
        LSTM(units=config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_MEDIUM_STACK (4 units, 1 dropout rates)
def LSTM_MEDIUM_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_MEDIUM_STACK', required_units=4, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(units=config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_HARD_STACK_BALANCED_DROPOUT (6 units, 5 dropout rates)
def LSTM_HARD_STACK_BALANCED_DROPOUT(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_HARD_STACK_BALANCED_DROPOUT', required_units=6, required_dropout_rates=5)
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
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# LSTM_HARD_STACK (6 units, 1 dropout rates)
def LSTM_HARD_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'LSTM_HARD_STACK', required_units=6, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(units=config["units"][4], return_sequences=True, name="LSTM_5"),
        LSTM(units=config["units"][5], return_sequences=False, name="LSTM_6"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])







## AutoRegressive Regression Models ##


