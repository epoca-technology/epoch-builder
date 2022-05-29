from keras import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Dropout, Flatten
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




# Classification DNN Stack 1
# C_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
def C_DNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S1', units=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(2, activation="softmax", name="Dense_2"),
    ])





# Classification DNN Stack 2
# C_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
def C_DNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S2', units=2, activations=2)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(2, activation="softmax", name="Dense_3"),
    ])




# Classification DNN Stack 2 with Dropout
# C_DNN_S2_DO
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def C_DNN_S2_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S2_DO', units=2, activations=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(2, activation="softmax", name="Dense_3"),
    ])






# Classification DNN Stack 3
# C_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
def C_DNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S3', units=3, activations=3)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(2, activation="softmax", name="Dense_4"),
    ])




# Classification DNN Stack 3 with Dropout
# C_DNN_S3_DO
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def C_DNN_S3_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S3_DO', units=3, activations=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(2, activation="softmax", name="Dense_4"),
    ])





# Classification DNN Stack 4
# C_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
def C_DNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S4', units=4, activations=4)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(2, activation="softmax", name="Dense_5"),
    ])





# Classification DNN Stack 4 with Dropout
# C_DNN_S4_DO
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def C_DNN_S4_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S4_DO', units=4, activations=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(2, activation="softmax", name="Dense_5"),
    ])






# Classification DNN Stack 5
# C_DNN_S5
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
def C_DNN_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S5', units=5, activations=5)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(m["units"][4], activation=m["activations"][4], name="Dense_5"),
        Dense(2, activation="softmax", name="Dense_6"),
    ])





# Classification DNN Stack 5 with Dropuot
# C_DNN_S5_DO
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def C_DNN_S5_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S5_DO', units=5, activations=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
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
        Dense(2, activation="softmax", name="Dense_6"),
    ])






# Classification DNN Stack 6
# C_DNN_S6
# 6 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5, Dense_6
# 6 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5, Dense_6
def C_DNN_S6(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S6', units=6, activations=6)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
        Dense(m["units"][0], activation=m["activations"][0], name="Dense_1"),
        Dense(m["units"][1], activation=m["activations"][1], name="Dense_2"),
        Dense(m["units"][2], activation=m["activations"][2], name="Dense_3"),
        Dense(m["units"][3], activation=m["activations"][3], name="Dense_4"),
        Dense(m["units"][4], activation=m["activations"][4], name="Dense_5"),
        Dense(m["units"][5], activation=m["activations"][5], name="Dense_6"),
        Dense(2, activation="softmax", name="Dense_7"),
    ])





# Classification DNN Stack 6 with Dropuot
# C_DNN_S6_DO
# 6 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5, Dense_6
# 6 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5, Dense_6
# 6 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5, Dropout_6
def C_DNN_S6_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_DNN_S6_DO', units=6, activations=6, dropout_rates=6)
    return Sequential([
        Input(shape=(m["features_num"],), name="Input_1"),
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
        Dense(m["units"][5], activation=m["activations"][5], name="Dense_6"),
        Dropout(m["dropout_rates"][5], name="Dropout_6"),
        Dense(2, activation="softmax", name="Dense_7"),
    ])














## Convolutional Neural Network ##





# Classification CNN Stack 1
# C_CNN_S1
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
def C_CNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S1', filters=1, activations=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 1 with MaxPooling
# C_CNN_S1_MP
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
# 1 pool_sizes:  MaxPooling1D_1
def C_CNN_S1_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S1_MP', filters=1, activations=1, pool_sizes=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 1 with MaxPooling and Dropout
# C_CNN_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 dropout_rates:  Dropout_1
def C_CNN_S1_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S1_MP_DO', filters=1, activations=1, pool_sizes=1, dropout_rates=1)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CNN Stack 2
# C_CNN_S2
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
def C_CNN_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S2', filters=2, activations=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])



# Classification CNN Stack 2 with Dropout
# C_CNN_S2_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def C_CNN_S2_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S2_DO', filters=2, activations=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 2 with MaxPooling
# C_CNN_S2_MP
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
# 2 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2
def C_CNN_S2_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S2_MP', filters=2, activations=2, pool_sizes=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 2 with MaxPooling and Dropout
# C_CNN_S2_MP_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def C_CNN_S2_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S2_MP_DO', filters=2, activations=2, pool_sizes=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CNN Stack 3
# C_CNN_S3
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
def C_CNN_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S3', filters=3, activations=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 3 with Dropout
# C_CNN_S3_DO
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def C_CNN_S3_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S3_DO', filters=3, activations=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CNN Stack 3 with MaxPooling
# C_CNN_S3_MP
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
def C_CNN_S3_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S3_MP', filters=3, activations=3, pool_sizes=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding='same', name="MaxPooling1D_3"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 3 with MaxPooling and Dropout
# C_CNN_S3_MP_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def C_CNN_S3_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S3_MP_DO', filters=3, activations=3, pool_sizes=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding='same', name="MaxPooling1D_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification CNN Stack 4
# C_CNN_S4
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
def C_CNN_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S4', filters=4, activations=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 4 with Dropout
# C_CNN_S4_DO
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def C_CNN_S4_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S4_DO', filters=4, activations=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 4 with MaxPooling
# C_CNN_S4_MP
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
def C_CNN_S4_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S4_MP', filters=4, activations=4, pool_sizes=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding='same', name="MaxPooling1D_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        MaxPooling1D(m["pool_sizes"][3], padding='same', name="MaxPooling1D_4"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CNN Stack 4 with MaxPooling and Dropout
# C_CNN_S4_MP_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def C_CNN_S4_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S4_MP_DO', filters=4, activations=4, pool_sizes=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding='same', name="MaxPooling1D_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        MaxPooling1D(m["pool_sizes"][3], padding='same', name="MaxPooling1D_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CNN Stack 5
# C_CNN_S5
# 5 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
def C_CNN_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S5', filters=5, activations=5)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        Conv1D(m["filters"][4], kernel_size=(int(m["features_num"]/5),), activation=m["activations"][4], padding='same', name="Conv1D_5"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification CNN Stack 5 with Dropout
# C_CNN_S5_DO
# 5 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def C_CNN_S5_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S5_DO', filters=5, activations=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Conv1D(m["filters"][4], kernel_size=(int(m["features_num"]/5),), activation=m["activations"][4], padding='same', name="Conv1D_5"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification CNN Stack 5 with MaxPooling
# C_CNN_S5_MP
# 5 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4, MaxPooling1D_5
def C_CNN_S5_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S5_MP', filters=5, activations=5, pool_sizes=5)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding='same', name="MaxPooling1D_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        MaxPooling1D(m["pool_sizes"][3], padding='same', name="MaxPooling1D_4"),
        Conv1D(m["filters"][4], kernel_size=(int(m["features_num"]/5),), activation=m["activations"][4], padding='same', name="Conv1D_5"),
        MaxPooling1D(m["pool_sizes"][4], padding='same', name="MaxPooling1D_5"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification CNN Stack 5 with MaxPooling and Dropout
# C_CNN_S5_MP_DO
# 5 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4, MaxPooling1D_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def C_CNN_S5_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CNN_S5_MP_DO', filters=5, activations=5, pool_sizes=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["features_num"],1), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Conv1D(m["filters"][1], kernel_size=(int(m["features_num"]/2),), activation=m["activations"][1], padding='same', name="Conv1D_2"),
        MaxPooling1D(m["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Conv1D(m["filters"][2], kernel_size=(int(m["features_num"]/3),), activation=m["activations"][2], padding='same', name="Conv1D_3"),
        MaxPooling1D(m["pool_sizes"][2], padding='same', name="MaxPooling1D_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Conv1D(m["filters"][3], kernel_size=(int(m["features_num"]/4),), activation=m["activations"][3], padding='same', name="Conv1D_4"),
        MaxPooling1D(m["pool_sizes"][3], padding='same', name="MaxPooling1D_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Conv1D(m["filters"][4], kernel_size=(int(m["features_num"]/5),), activation=m["activations"][4], padding='same', name="Conv1D_5"),
        MaxPooling1D(m["pool_sizes"][4], padding='same', name="MaxPooling1D_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Flatten(name="Flatten_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])

























## Long Short-Term Memory Recurrent Neural Network ##




# Classification LSTM Stack 1
# C_LSTM_S1
# 1 units: LSTM_1
def C_LSTM_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S1', units=1)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])




# Classification LSTM Stack 2
# C_LSTM_S2
# 2 units: LSTM_1, LSTM_2
def C_LSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S2', units=2)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], name="LSTM_2"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification LSTM Stack 2 with Dropout
# C_LSTM_S2_DO
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def C_LSTM_S2_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S2_DO', units=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification LSTM Stack 3
# C_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
def C_LSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S3', units=3)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], name="LSTM_3"),
        Dense(2, activation='softmax', name="Dense_1")
    ])




# Classification LSTM Stack 3 with Dropout
# C_LSTM_S3_DO
# 3 units: LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def C_LSTM_S3_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S3_DO', units=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification LSTM Stack 4
# C_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
def C_LSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S4', units=4)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], name="LSTM_4"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification LSTM Stack 4 with Dropout
# C_LSTM_S4_DO
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def C_LSTM_S4_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S4_DO', units=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification LSTM Stack 5
# C_LSTM_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_4, LSTM_5
def C_LSTM_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S5', units=5)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], name="LSTM_5"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification LSTM Stack 5 with Dropout
# C_LSTM_S5_DO
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def C_LSTM_S5_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S5_DO', units=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        LSTM(m["units"][4], name="LSTM_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification LSTM Stack 6
# C_LSTM_S6
# 6 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_4, LSTM_5, LSTM_6
def C_LSTM_S6(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S6', units=6)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], return_sequences=True, name="LSTM_5"),
        LSTM(m["units"][5], name="LSTM_6"),
        Dense(2, activation='softmax', name="Dense_1")
    ])




# Classification LSTM Stack 6 with Dropout
# C_LSTM_S6_DO
# 6 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5, LSTM_6
# 6 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5, Dropout_6
def C_LSTM_S6_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_LSTM_S6_DO', units=6, dropout_rates=6)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        LSTM(m["units"][4], return_sequences=True, name="LSTM_5"),
        Dropout(m["dropout_rates"][4], name="Dropout_5"),
        LSTM(m["units"][5], name="LSTM_6"),
        Dropout(m["dropout_rates"][5], name="Dropout_6"),
        Dense(2, activation='softmax', name="Dense_1")
    ])












## Convolutional Long Short-Term Memory Recurrent Neural Network ##





# Classification CLSTM Stack 1
# C_CLSTM_S1
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 units:          LSTM_1
def C_CLSTM_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S1', filters=1, activations=1, units=1)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])




# Classification CLSTM Stack 1 with Dropout
# C_CLSTM_S1_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 units:          LSTM_1
# 1 dropout_rates:  Dropout_1
def C_CLSTM_S1_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S1_DO', filters=1, activations=1, units=1, dropout_rates=1)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 1 with MaxPooling
# C_CLSTM_S1_MP
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
def C_CLSTM_S1_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S1_MP', filters=1, activations=1, pool_sizes=1, units=1)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 1 with MaxPooling and Dropout
# C_CLSTM_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 dropout_rates:  Dropout_1
def C_CLSTM_S1_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S1_MP_DO', filters=1, activations=1, pool_sizes=1, units=1, dropout_rates=1)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=False, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        Dense(2, activation='softmax', name="Dense_1")
    ])








# Classification CLSTM Stack 2
# C_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 2 units:          LSTM_1, LSTM_2
def C_CLSTM_S2(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S2', filters=1, activations=1, units=2)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 2 with Dropout
# C_CLSTM_S2_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def C_CLSTM_S2_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S2_DO', filters=1, activations=1, units=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 2 with MaxPooling
# C_CLSTM_S2_MP
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
def C_CLSTM_S2_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S2_MP', filters=1, activations=1, pool_sizes=1, units=2)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 2 with MaxPooling and Dropout
# C_CLSTM_S2_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
def C_CLSTM_S2_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S2_MP_DO', filters=1, activations=1, pool_sizes=1, units=2, dropout_rates=2)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=False, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 3
# C_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
def C_CLSTM_S3(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S3', filters=1, activations=1, units=3)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 3 with Dropout
# C_CLSTM_S3_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def C_CLSTM_S3_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S3_DO', filters=1, activations=1, units=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 3 with MaxPooling
# C_CLSTM_S3_MP
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
def C_CLSTM_S3_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S3_MP', filters=1, activations=1, pool_sizes=1, units=3)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 3 with MaxPooling and Dropout
# C_CLSTM_S3_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
def C_CLSTM_S3_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S3_MP_DO', filters=1, activations=1, pool_sizes=1, units=3, dropout_rates=3)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=False, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 4
# C_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
def C_CLSTM_S4(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S4', filters=1, activations=1, units=4)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 4 with Dropout
# C_CLSTM_S4_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def C_CLSTM_S4_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S4_DO', filters=1, activations=1, units=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(2, activation='softmax', name="Dense_1")
    ])






# Classification CLSTM Stack 4 with MaxPooling
# C_CLSTM_S4_MP
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
def C_CLSTM_S4_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S4_MP', filters=1, activations=1, pool_sizes=1, units=4)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 4 with MaxPooling and Dropout
# C_CLSTM_S4_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
def C_CLSTM_S4_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S4_MP_DO', filters=1, activations=1, pool_sizes=1, units=4, dropout_rates=4)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(m["dropout_rates"][0], name="Dropout_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(m["dropout_rates"][1], name="Dropout_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        Dropout(m["dropout_rates"][2], name="Dropout_3"),
        LSTM(m["units"][3], return_sequences=False, name="LSTM_4"),
        Dropout(m["dropout_rates"][3], name="Dropout_4"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification CLSTM Stack 5
# C_CLSTM_S5
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
def C_CLSTM_S5(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S5', filters=1, activations=1, units=5)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dense(2, activation='softmax', name="Dense_1")
    ])





# Classification CLSTM Stack 5 with Dropout
# C_CLSTM_S5_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def C_CLSTM_S5_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S5_DO', filters=1, activations=1, units=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
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
        Dense(2, activation='softmax', name="Dense_1")
    ])




# Classification CLSTM Stack 5 with MaxPooling
# C_CLSTM_S5_MP
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
def C_CLSTM_S5_MP(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S5_MP', filters=1, activations=1, pool_sizes=1, units=5)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        LSTM(m["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(m["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(m["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(m["units"][3], return_sequences=True, name="LSTM_4"),
        LSTM(m["units"][4], return_sequences=False, name="LSTM_5"),
        Dense(2, activation='softmax', name="Dense_1")
    ])







# Classification CLSTM Stack 5 with MaxPooling and Dropout
# C_CLSTM_S5_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
def C_CLSTM_S5_MP_DO(m: IKerasModelConfig) -> Sequential:
    validate(m, 'C_CLSTM_S5_MP_DO', filters=1, activations=1, pool_sizes=1, units=5, dropout_rates=5)
    return Sequential([
        Input(shape=(m["features_num"],1,), name="Input_1"),
        Conv1D(m["filters"][0], kernel_size=(m["features_num"],), activation=m["activations"][0], name="Conv1D_1"),
        MaxPooling1D(m["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
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
        Dense(2, activation='softmax', name="Dense_1")
    ])