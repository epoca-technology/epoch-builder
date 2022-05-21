from tensorflow import zeros
from keras import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv1D, MaxPooling1D, LSTM, Dropout
from modules.keras_models import IKerasModelConfig, validate





## UNIT TEST MODEL ##

# R_UNIT_TEST 
# 1 units:       Dense_1
# 1 activations: Dense_1
def R_UNIT_TEST(m: IKerasModelConfig) -> Sequential:
    validate(m, 'R_UNIT_TEST', units=1, activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(m["units"][0], activation=m["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(m['predictions'], kernel_initializer=zeros, name="Dense_2"),
        Flatten(name="Flatten_1")
    ])






## TRADITIONAL REGRESSION MODELS ##





## Deep Neural Network ##





# Regression DNN Stack 1
# R_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
def R_DNN_S1(m: IKerasModelConfig) -> Sequential:
    validate(m, 'R_DNN_S1', units=1, activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(m["units"][0], activation=m["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(m['predictions'], kernel_initializer=zeros, name="Dense_2"),
        Flatten(name="Flatten_1")
    ])





# Regression DNN Stack 2
# R_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
def R_DNN_S2(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_DNN_S2', units=2, activations=2)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config["units"][1], activation=config["activations"][1], kernel_initializer="normal", name="Dense_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_3"),
        Flatten(name="Flatten_1")
    ])




# Regression DNN Stack 3
# R_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
def R_DNN_S3(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_DNN_S3', units=3, activations=3)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config["units"][1], activation=config["activations"][1], kernel_initializer="normal", name="Dense_2"),
        Dense(config["units"][2], activation=config["activations"][2], kernel_initializer="normal", name="Dense_3"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_4"),
        Flatten(name="Flatten_1")
    ])




# Regression DNN Stack 4
# R_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
def R_DNN_S4(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_DNN_S4', units=4, activations=4)
    return Sequential([
        Lambda(lambda x: x[:, -1:, :], name="Lambda_1"),
        Dense(config["units"][0], activation=config["activations"][0], kernel_initializer="normal", name="Dense_1"),
        Dense(config["units"][1], activation=config["activations"][1], kernel_initializer="normal", name="Dense_2"),
        Dense(config["units"][2], activation=config["activations"][2], kernel_initializer="normal", name="Dense_3"),
        Dense(config["units"][3], activation=config["activations"][3], kernel_initializer="normal", name="Dense_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_5"),
        Flatten(name="Flatten_1")
    ])





# Regression DNN Stack 5
# R_DNN_S5
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
def R_DNN_S5(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_DNN_S5', units=5, activations=5)
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





# Regression CNN Stack 1
# R_CNN_S1
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
def R_CNN_S1(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_CNN_S1', filters=1, activations=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"],), name="Conv1D_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])





# Regression CNN Stack 1 with MaxPooling
# R_CNN_S1_MP
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
# 1 pool_sizes:  MaxPooling1D_1
def R_CNN_S1_MP(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_CNN_S1_MP', filters=1, activations=1, pool_sizes=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"],), name="Conv1D_1"),
        MaxPooling1D(config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])





# Regression CNN Stack 2
# R_CNN_S2
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
def R_CNN_S2(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_CNN_S2', filters=2, activations=2)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"],), name="Conv1D_1"),
        Conv1D(config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(int(config["lookback"]/2),), name="Conv1D_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])





# Regression CNN Stack 2 with MaxPooling
# R_CNN_S2_MP
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
# 2 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2
def R_CNN_S2_MP(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_CNN_S2_MP', filters=2, activations=2, pool_sizes=2)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Conv1D(config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        MaxPooling1D(config["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])





# Regression CNN Stack 2 with MaxPooling and Dropout
# R_CNN_S2_MP_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 dropout_rates:  Dropout_1
def R_CNN_S2_MP_DO(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_CNN_S2_MP_DO', filters=2, activations=2, pool_sizes=2, dropout_rates=1)
    return Sequential([
        Lambda(lambda x: x[:, -config["lookback"]:, :], name="Lambda_1"),
        Conv1D(config["filters"][0], activation=config["activations"][0], kernel_size=(config["lookback"]), name="Conv1D_1"),
        MaxPooling1D(config["pool_sizes"][0], padding='same', name="MaxPooling1D_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        Conv1D(config["filters"][1], activation=config["activations"][1], padding='same', kernel_size=(config["lookback"]), name="Conv1D_2"),
        MaxPooling1D(config["pool_sizes"][1], padding='same', name="MaxPooling1D_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
        Flatten(name="Flatten_1")
    ])













## Long Short-Term Memory Recurrent Neural Network ##



# Regression LSTM Stack 1
# R_LSTM_S1
# 1 units: LSTM_1
def R_LSTM_S1(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S1', units=1)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# Regression LSTM Stack 2
# R_LSTM_S2
# 2 units: LSTM_1, LSTM_2
def R_LSTM_S2(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S2', units=2)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])




# Regression LSTM Stack 2 with Dropout
# R_LSTM_S2_DO
# 2 units:          LSTM_1, LSTM_2
# 1 dropout rates:  Dropout_1
def R_LSTM_S2_DO(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S2_DO', units=2, dropout_rates=1)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][1], return_sequences=False, name="LSTM_2"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# Regression LSTM Stack 3
# R_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
def R_LSTM_S3(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S3', units=3)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(config["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# Regression LSTM Stack 3 with Dropout
# R_LSTM_S3_DO
# 3 units: LSTM_1, LSTM_2, LSTM_3
# 2 dropout rates:  Dropout_1, Dropout_2
def R_LSTM_S3_DO(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S3_DO', units=3, dropout_rates=2)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][1], name="Dropout_2"),
        LSTM(config["units"][2], return_sequences=False, name="LSTM_3"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])




# Regression LSTM Stack 4
# R_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
def R_LSTM_S4(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S4', units=4)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        LSTM(config["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# Regression LSTM Stack 4 with Dropout
# R_LSTM_S4_DO
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 dropout rates:  Dropout_1
def R_LSTM_S4_DO(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S4_DO', units=4, dropout_rates=1)
    return Sequential([
        LSTM(config["units"][0], return_sequences=True, name="LSTM_1"),
        LSTM(config["units"][1], return_sequences=True, name="LSTM_2"),
        Dropout(config["dropout_rates"][0], name="Dropout_1"),
        LSTM(config["units"][2], return_sequences=True, name="LSTM_3"),
        LSTM(config["units"][3], return_sequences=False, name="LSTM_4"),
        Dense(config['predictions'], kernel_initializer=zeros, name="Dense_1"),
    ])



# Regression LSTM Stack 4 with Balanced Dropout
# R_LSTM_S4_BDO
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 3 dropout rates:  Dropout_1, Dropout_2, Dropout_3
def R_LSTM_S4_BDO(config: IKerasModelConfig) -> Sequential:
    validate(config, 'R_LSTM_S4_BDO', units=4, dropout_rates=1)
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