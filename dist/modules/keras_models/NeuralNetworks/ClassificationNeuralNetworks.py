from keras import Sequential
from keras.layers import Input, Flatten, Reshape, Dense, Conv1D, MaxPooling1D, LSTM
from modules.types import IKerasModelConfig
from modules.keras_models.NeuralNetworks.validator import validate






#####################
## UNIT TEST MODEL ##
#####################


# C_UNIT_TEST 
# 1 units:       Dense_1
# 1 activations: Dense_1
def C_UNIT_TEST(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_UNIT_TEST', units=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])










#########################
## Deep Neural Network ##
#########################




# Classification DNN Stack 1
# C_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
def C_DNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_DNN_S1", units=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])






# Classification DNN Stack 2
# C_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
def C_DNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_DNN_S2", units=2, activations=2)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])









# Classification DNN Stack 3
# C_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
def C_DNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_DNN_S3", units=3, activations=3)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification DNN Stack 4
# C_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
def C_DNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_DNN_S4", units=4, activations=4)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])















##################################
## Convolutional Neural Network ##
##################################






# Classification CNN Stack 1
# C_CNN_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 activations:    Conv1D_1
def C_CNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S1", filters=1, kernel_sizes=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])




# Classification CNN Stack 1 with MaxPooling
# C_CNN_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 activations:    Conv1D_1
def C_CNN_S1_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S1_MP", filters=1, kernel_sizes=1, pool_sizes=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])









# Classification CNN Stack 2
# C_CNN_S2
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
def C_CNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S2", filters=2, kernel_sizes=2, activations=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(m["kernel_sizes"][1],), activation=m["activations"][1], padding="same", name="Conv1D_2"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification CNN Stack 2 with MaxPooling
# C_CNN_S2_MP
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 activations:    Conv1D_1, Conv1D_2
def C_CNN_S2_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S2_MP", filters=2, kernel_sizes=2, pool_sizes=2, activations=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding="same", name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(m["kernel_sizes"][1],), activation=m["activations"][1], padding="same", name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding="same", name="MaxPooling1D_2"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification CNN Stack 3
# C_CNN_S3
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
def C_CNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S3", filters=3, kernel_sizes=3, activations=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(m["kernel_sizes"][1],), activation=m["activations"][1], padding="same", name="Conv1D_2"),
        Conv1D(m["filters"][2], kernel_size=(m["kernel_sizes"][2],), activation=m["activations"][2], padding="same", name="Conv1D_3"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification CNN Stack 3 with MaxPooling
# C_CNN_S3_MP
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
def C_CNN_S3_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S3_MP", filters=3, kernel_sizes=3, pool_sizes=3, activations=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding="same", name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(m["kernel_sizes"][1],), activation=m["activations"][1], padding="same", name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding="same", name="MaxPooling1D_2"),
        Conv1D(m["filters"][2], kernel_size=(m["kernel_sizes"][2],), activation=m["activations"][2], padding="same", name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding="same", name="MaxPooling1D_3"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])









# Classification CNN Stack 4
# C_CNN_S4
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
def C_CNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S4", filters=4, kernel_sizes=4, activations=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(m["kernel_sizes"][1],), activation=m["activations"][1], padding="same", name="Conv1D_2"),
        Conv1D(m["filters"][2], kernel_size=(m["kernel_sizes"][2],), activation=m["activations"][2], padding="same", name="Conv1D_3"),
        Conv1D(m["filters"][3], kernel_size=(m["kernel_sizes"][3],), activation=m["activations"][3], padding="same", name="Conv1D_4"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])









# Classification CNN Stack 4 with MaxPooling
# C_CNN_S4_MP
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
def C_CNN_S4_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CNN_S4_MP", filters=4, kernel_sizes=4, pool_sizes=4, activations=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding="same", name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(m["kernel_sizes"][1],), activation=m["activations"][1], padding="same", name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding="same", name="MaxPooling1D_2"),
        Conv1D(m["filters"][2], kernel_size=(m["kernel_sizes"][2],), activation=m["activations"][2], padding="same", name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding="same", name="MaxPooling1D_3"),
        Conv1D(m["filters"][3], kernel_size=(m["kernel_sizes"][3],), activation=m["activations"][3], padding="same", name="Conv1D_4"),
        MaxPooling1D(m["pool_sizes"][3], padding="same", name="MaxPooling1D_4"),
        Flatten(name="Flatten_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])













#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################








# Classification LSTM Stack 1
# C_LSTM_S1
# 1 units: LSTM_1
def C_LSTM_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_LSTM_S1", units=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])





# Classification LSTM Stack 2
# C_LSTM_S2
# 2 units: LSTM_1, LSTM_2
def C_LSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_LSTM_S2", units=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification LSTM Stack 3
# C_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
def C_LSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_LSTM_S3", units=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification LSTM Stack 4
# C_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
def C_LSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_LSTM_S4", units=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])
















###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################




# Classification CLSTM Stack 1
# C_CLSTM_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
def C_CLSTM_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S1", filters=1, kernel_sizes=1, units=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification CLSTM Stack 1 with MaxPooling
# C_CLSTM_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
def C_CLSTM_S1_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S1_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])









# Classification CLSTM Stack 2
# C_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
def C_CLSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S2", filters=1, kernel_sizes=1, units=2, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])











# Classification CLSTM Stack 2 with MaxPooling
# C_CLSTM_S2_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
def C_CLSTM_S2_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S2_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])










# Classification CLSTM Stack 3
# C_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
def C_CLSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S3", filters=1, kernel_sizes=1, units=3, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])










# Classification CLSTM Stack 3 with MaxPooling
# C_CLSTM_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
def C_CLSTM_S3_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S3_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])









# Classification CLSTM Stack 4
# C_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
def C_CLSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S4", filters=1, kernel_sizes=1, units=4, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])










# Classification CLSTM Stack 4 with MaxPooling
# C_CLSTM_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
def C_CLSTM_S4_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, "C_CLSTM_S4_MP", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Reshape((m["features_num"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(2, activation="softmax", name="Dense_Output")
    ])

