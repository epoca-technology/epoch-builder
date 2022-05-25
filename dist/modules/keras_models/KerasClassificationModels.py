from keras import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from modules.keras_models import IKerasModelConfig, validate





## UNIT TEST MODEL ##


# C_UNIT_TEST 
# 1 units:       Dense_1
# 1 activations: Dense_1
def C_UNIT_TEST(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_UNIT_TEST', units=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(2, activation="softmax", name="Dense_2"),
    ])








## Deep Neural Network ##




## Convolutional Neural Network ##





## Long Short-Term Memory Recurrent Neural Network ##



