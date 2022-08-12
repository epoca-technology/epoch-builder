from keras import Sequential
from keras.layers import Input, Flatten, Reshape, Dense, Conv1D, MaxPooling1D, LSTM
from modules.types import IKerasModelConfig
from modules.keras_models.NeuralNetworks.validator import validate






#####################
## UNIT TEST MODEL ##
#####################


# R_UNIT_TEST 
# 1 units:       Dense_1
# 1 activations: Dense_1
def R_UNIT_TEST(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_UNIT_TEST", units=1, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])










#########################
## Deep Neural Network ##
#########################




# Regression DNN Stack 1
# R_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
def R_DNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_DNN_S1", units=1, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])






# Regression DNN Stack 2
# R_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
def R_DNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_DNN_S2", units=2, activations=2)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])










# Regression DNN Stack 3
# R_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
def R_DNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_DNN_S3", units=3, activations=3)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])








# Regression DNN Stack 4
# R_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
def R_DNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_DNN_S4", units=4, activations=4)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])















##################################
## Convolutional Neural Network ##
##################################






# Regression CNN Stack 1
# R_CNN_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          Dense_1
# 2 activations:    Conv1D_1, Dense_1
def R_CNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S1", filters=1, kernel_sizes=1, units=1, activations=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])




# Regression CNN Stack 1 with MaxPooling
# R_CNN_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          Dense_1
# 2 activations:    Conv1D_1, Dense_1
def R_CNN_S1_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S1_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=1, activations=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])









# Regression CNN Stack 2
# R_CNN_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
def R_CNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S2", filters=1, kernel_sizes=1, units=2, activations=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])






# Regression CNN Stack 2 with MaxPooling
# R_CNN_S2_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
def R_CNN_S2_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S2_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])







# Regression CNN Stack 3
# R_CNN_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
def R_CNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S3", filters=1, kernel_sizes=1, units=3, activations=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])




# Regression CNN Stack 3 with MaxPooling
# R_CNN_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
def R_CNN_S3_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S3_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])














# Regression CNN Stack 4
# R_CNN_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
def R_CNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S4", filters=1, kernel_sizes=1, units=4, activations=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])







# Regression CNN Stack 4 with MaxPooling
# R_CNN_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
def R_CNN_S4_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CNN_S4_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Flatten(name="Flatten_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])
















#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################








# Regression LSTM Stack 1
# R_LSTM_S1
# 1 units: LSTM_1
def R_LSTM_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_LSTM_S1", units=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])





# Regression LSTM Stack 2
# R_LSTM_S2
# 2 units: LSTM_1, LSTM_2
def R_LSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_LSTM_S2", units=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])









# Regression LSTM Stack 3
# R_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
def R_LSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_LSTM_S3", units=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])








# Regression LSTM Stack 4
# R_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
def R_LSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_LSTM_S4", units=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])
















###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################




# Regression CLSTM Stack 1
# R_CLSTM_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
def R_CLSTM_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S1", filters=1, kernel_sizes=1, units=1, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])




# Regression CLSTM Stack 1 with MaxPooling
# R_CLSTM_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
def R_CLSTM_S1_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S1_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=1, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])










# Regression CLSTM Stack 2
# R_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
def R_CLSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S2", filters=1, kernel_sizes=1, units=2, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 2 with MaxPooling
# R_CLSTM_S2_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
def R_CLSTM_S2_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S2_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])












# Regression CLSTM Stack 3
# R_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
def R_CLSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S3", filters=1, kernel_sizes=1, units=3, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])






# Regression CLSTM Stack 3 with MaxPooling
# R_CLSTM_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
def R_CLSTM_S3_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S3_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])











# Regression CLSTM Stack 4
# R_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
def R_CLSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S4", filters=1, kernel_sizes=1, units=4, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])






# Regression CLSTM Stack 4 with MaxPooling
# R_CLSTM_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
def R_CLSTM_S4_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "R_CLSTM_S4_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(1 if m["autoregressive"] else m["predictions"], name="Dense_Output")
    ])


