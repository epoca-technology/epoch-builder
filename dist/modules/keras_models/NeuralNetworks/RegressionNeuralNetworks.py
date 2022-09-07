from keras import Sequential
from keras.layers import Input, Reshape, Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from modules._types import IKerasModelConfig
from modules.keras_models.NeuralNetworks.validator import validate




#########################
## Deep Neural Network ##
#########################





# Regression DNN Stack 2
# KR_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
def KR_DNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_S2", units=2, activations=2)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression DNN Stack 2 with Dropout
# KR_DNN_DO_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def KR_DNN_DO_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_DO_S2", units=2, activations=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression DNN Stack 3
# KR_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
def KR_DNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_S3", units=3, activations=3)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression DNN Stack 3 with Dropout
# KR_DNN_DO_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def KR_DNN_DO_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_DO_S3", units=3, activations=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression DNN Stack 4
# KR_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
def KR_DNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_S4", units=4, activations=4)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])



# Regression DNN Stack 4 with Dropout
# KR_DNN_DO_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def KR_DNN_DO_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_DO_S4", units=4, activations=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression DNN Stack 5
# KR_DNN_S5
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
def KR_DNN_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_S5", units=5, activations=5)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(m["units"][4], activation=m["activations"][4], name="Dense_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression DNN Stack 5 with Dropout
# KR_DNN_DO_S5
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def KR_DNN_DO_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_DNN_DO_S5", units=5, activations=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["lookback"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["units"][4], activation=m["activations"][4], name="Dense_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])











##################################
## Convolutional Neural Network ##
##################################







# Regression CNN Stack 2
# KR_CNN_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
def KR_CNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_S2", filters=1, kernel_sizes=1, units=2, activations=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CNN Stack 2 with Dropout
# KR_CNN_DO_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def KR_CNN_DO_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_DO_S2", filters=1, kernel_sizes=1, units=2, activations=3, dropout_rates=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 2 with MaxPooling
# KR_CNN_MP_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
def KR_CNN_MP_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_S2", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 2 with MaxPooling and Dropout
# KR_CNN_MP_DO_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def KR_CNN_MP_DO_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_DO_S2", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=3, dropout_rates=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CNN Stack 3
# KR_CNN_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
def KR_CNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_S3", filters=1, kernel_sizes=1, units=3, activations=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])






# Regression CNN Stack 3 with Dropout
# KR_CNN_DO_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def KR_CNN_DO_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_DO_S3", filters=1, kernel_sizes=1, units=3, activations=4, dropout_rates=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])






# Regression CNN Stack 3 with MaxPooling
# KR_CNN_MP_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
def KR_CNN_MP_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_S3", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 3 with MaxPooling and Dropout
# KR_CNN_MP_DO_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def KR_CNN_MP_DO_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_DO_S3", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=4, dropout_rates=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])






# Regression CNN Stack 4
# KR_CNN_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
def KR_CNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_S4", filters=1, kernel_sizes=1, units=4, activations=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 4 with Dropout
# KR_CNN_DO_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def KR_CNN_DO_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_DO_S4", filters=1, kernel_sizes=1, units=4, activations=5, dropout_rates=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 4 with MaxPooling
# KR_CNN_MP_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
def KR_CNN_MP_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_S4", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=5)
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
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CNN Stack 4 with MaxPooling and Dropout
# KR_CNN_MP_DO_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def KR_CNN_MP_DO_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_DO_S4", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=5, dropout_rates=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])






# Regression CNN Stack 5
# KR_CNN_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
def KR_CNN_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_S5", filters=1, kernel_sizes=1, units=5, activations=6)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Dense(m["units"][4], activation=m["activations"][5], name="Dense_5"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 5 with Dropout
# KR_CNN_DO_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def KR_CNN_DO_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_DO_S5", filters=1, kernel_sizes=1, units=5, activations=6, dropout_rates=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["units"][4], activation=m["activations"][5], name="Dense_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CNN Stack 5 with MaxPooling
# KR_CNN_MP_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
def KR_CNN_MP_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_S5", filters=1, kernel_sizes=1, pool_sizes=1, units=5, activations=6)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Dense(m["units"][4], activation=m["activations"][5], name="Dense_5"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CNN Stack 5 with MaxPooling and Dropout
# KR_CNN_MP_DO_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def KR_CNN_MP_DO_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CNN_MP_DO_S5", filters=1, kernel_sizes=1, pool_sizes=1, units=5, activations=6, dropout_rates=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        Dense(m["units"][0], activation=m["activations"][1], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][2], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][3], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][4], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["units"][4], activation=m["activations"][5], name="Dense_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Flatten(name="Flatten_1"),
        Dense(m["predictions"], name="Dense_Output")
    ])












#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################





# Regression LSTM Stack 2
# KR_LSTM_S2
# 2 units: LSTM_1, LSTM_2
def KR_LSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_S2", units=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression LSTM Stack 2 with Dropout
# KR_LSTM_DO_S2
# 2 units: LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def KR_LSTM_DO_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_DO_S2", units=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression LSTM Stack 3
# KR_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
def KR_LSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_S3", units=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression LSTM Stack 3 with Dropout
# KR_LSTM_DO_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def KR_LSTM_DO_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_DO_S3", units=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression LSTM Stack 4
# KR_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
def KR_LSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_S4", units=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression LSTM Stack 4 with Dropout
# KR_LSTM_DO_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def KR_LSTM_DO_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_DO_S4", units=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression LSTM Stack 5
# KR_LSTM_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
def KR_LSTM_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_S5", units=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression LSTM Stack 5 with Dropout
# KR_LSTM_DO_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def KR_LSTM_DO_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_LSTM_DO_S5", units=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])














###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################







# Regression CLSTM Stack 2
# KR_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
def KR_CLSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_S2", filters=1, kernel_sizes=1, units=2, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CLSTM Stack 2 with Dropout
# KR_CLSTM_DO_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
# 2 dropout_rates:  Dropout_1, Dropout_2
def KR_CLSTM_DO_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_DO_S2", filters=1, kernel_sizes=1, units=2, activations=1, dropout_rates=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 2 with MaxPooling
# KR_CLSTM_MP_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
def KR_CLSTM_MP_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_S2", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])



# Regression CLSTM Stack 2 with MaxPooling and Dropout
# KR_CLSTM_MP_DO_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
# 2 dropout_rates:  Dropout_1, Dropout_2
def KR_CLSTM_MP_DO_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_DO_S2", filters=1, kernel_sizes=1, pool_sizes=1, units=2, activations=1, dropout_rates=2)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 3
# KR_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
def KR_CLSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_S3", filters=1, kernel_sizes=1, units=3, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CLSTM Stack 3 with Dropout
# KR_CLSTM_DO_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def KR_CLSTM_DO_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_DO_S3", filters=1, kernel_sizes=1, units=3, activations=1, dropout_rates=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 3 with MaxPooling
# KR_CLSTM_MP_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
def KR_CLSTM_MP_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_S3", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 3 with MaxPooling and Dropout
# KR_CLSTM_MP_DO_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def KR_CLSTM_MP_DO_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_DO_S3", filters=1, kernel_sizes=1, pool_sizes=1, units=3, activations=1, dropout_rates=3)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 4
# KR_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
def KR_CLSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_S4", filters=1, kernel_sizes=1, units=4, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])




# Regression CLSTM Stack 4 with Dropout
# KR_CLSTM_DO_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def KR_CLSTM_DO_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_DO_S4", filters=1, kernel_sizes=1, units=4, activations=1, dropout_rates=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 4 with MaxPooling
# KR_CLSTM_MP_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
def KR_CLSTM_MP_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_S4", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 4 with MaxPooling and Dropout
# KR_CLSTM_MP_DO_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def KR_CLSTM_MP_DO_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_DO_S4", filters=1, kernel_sizes=1, pool_sizes=1, units=4, activations=1, dropout_rates=4)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 5
# KR_CLSTM_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
def KR_CLSTM_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_S5", filters=1, kernel_sizes=1, units=5, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 5 with Dropout
# KR_CLSTM_DO_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def KR_CLSTM_DO_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_DO_S5", filters=1, kernel_sizes=1, units=5, activations=1, dropout_rates=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])






# Regression CLSTM Stack 5 with MaxPooling
# KR_CLSTM_MP_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
def KR_CLSTM_MP_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_S5", filters=1, kernel_sizes=1, pool_sizes=1, units=5, activations=1)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])





# Regression CLSTM Stack 5 with MaxPooling and Dropout
# KR_CLSTM_MP_DO_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def KR_CLSTM_MP_DO_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, "KR_CLSTM_MP_DO_S5", filters=1, kernel_sizes=1, pool_sizes=1, units=5, activations=1, dropout_rates=5)
    return Sequential([
        Input(shape=(m["lookback"],1), name="Input_1"),
        Reshape((m["lookback"],1,), name="Reshape_1"),
        Conv1D(m["filters"][0], kernel_size=(m["kernel_sizes"][0],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Dense(m["predictions"], name="Dense_Output")
    ])