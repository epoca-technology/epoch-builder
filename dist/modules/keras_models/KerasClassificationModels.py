from typing import Union
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from modules.keras_models import IKerasModelConfig, validate




## Dense Neural Network ##




## Convolutional Neural Network ##





## Long Short-Term Memory Recurrent Neural Network ##




# LSTM_HARD_STACK (6 units, 1 dropout rates)
def LSTM_HARD_STACK(config: IKerasModelConfig) -> Sequential:
    validate(config, 'classification', 'LSTM_HARD_STACK', required_units=6, required_dropout_rates=1)
    return Sequential([
        LSTM(units=config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(units=config["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(units=config["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(units=config["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(units=config["units"][4], return_sequences=True, name="LSTM_5"),
        LSTM(units=config["units"][5], return_sequences=False, name="LSTM_6"),
        Dense(units=2, activation='softmax')
    ])