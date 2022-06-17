from keras import Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout
from modules.keras_models import IKerasModelConfig, validate







## UNIT TEST MODEL ##



# R_UNIT_TEST 
# 1 units:       Dense_1
# 1 activations: Dense_1
def R_UNIT_TEST(m: IKerasModelConfig) -> Sequential:
    validate(m, 'R_UNIT_TEST', units=1, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])



