from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


## Long Short-Term Memory Recurrent Neural Network ##


# Network Type
class ILSTM(TypedDict):
    C_LSTM_S1: List[IKerasModelConfig]
    C_LSTM_S2: List[IKerasModelConfig]
    C_LSTM_S2_DO: List[IKerasModelConfig]
    C_LSTM_S3: List[IKerasModelConfig]
    C_LSTM_S3_DO: List[IKerasModelConfig]
    C_LSTM_S4: List[IKerasModelConfig]
    C_LSTM_S4_DO: List[IKerasModelConfig]





# Classification LSTM Stack 1
# C_LSTM_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
C_LSTM_S1: List[IKerasModelConfig] = [
    {"units": [16]},
    {"units": [32]},
    {"units": [64]},
    {"units": [128]}
]




# Classification LSTM Stack 2
# C_LSTM_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
C_LSTM_S2: List[IKerasModelConfig] = [
    {"units": [16, 16]},

    {"units": [32, 16]},
    {"units": [16, 32]},
    {"units": [32, 32]},

    {"units": [64, 32]},
    {"units": [32, 64]},
    {"units": [64, 64]},
    
    {"units": [16, 64]},
    {"units": [64, 16]},

    {"units": [64, 128]},
    {"units": [128, 64]},
    {"units": [128, 128]},

    {"units": [16, 128]},
    {"units": [128, 16]},
    {"units": [32, 128]},
    {"units": [128, 32]},
]





# Classification LSTM Stack 2 with Dropout
# C_LSTM_S2_DO
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_LSTM_S2_DO: List[IKerasModelConfig] = [
        {"units": [16, 16], "dropout_rates": [0, 0]},

        {"units": [32, 16], "dropout_rates": [0, 0]},
        {"units": [16, 32], "dropout_rates": [0, 0]},
        {"units": [32, 32], "dropout_rates": [0, 0]},

        {"units": [64, 32], "dropout_rates": [0, 0]},
        {"units": [32, 64], "dropout_rates": [0, 0]},
        {"units": [64, 64], "dropout_rates": [0, 0]},
        
        {"units": [64, 16], "dropout_rates": [0, 0]},

        {"units": [64, 128], "dropout_rates": [0, 0]},
        {"units": [128, 64], "dropout_rates": [0, 0]},
        {"units": [128, 128], "dropout_rates": [0, 0]},

        {"units": [128, 16], "dropout_rates": [0, 0]},
    ]



# Classification LSTM Stack 3
# C_LSTM_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
C_LSTM_S3: List[IKerasModelConfig] = [
    {"units": [16, 16, 16]},

    {"units": [32, 16, 16]},
    {"units": [16, 32, 16]},
    {"units": [16, 16, 32]},
    {"units": [32, 32, 16]},
    {"units": [16, 32, 32]},
    {"units": [32, 32, 32]},

    {"units": [64, 32, 32]},
    {"units": [32, 64, 32]},
    {"units": [32, 32, 64]},
    {"units": [64, 64, 32]},
    {"units": [32, 64, 64]},
    {"units": [64, 64, 64]},

    {"units": [64, 32, 16]},

    {"units": [128, 64, 64]},
    {"units": [64, 128, 64]},
    {"units": [64, 64, 128]},
    {"units": [128, 128, 64]},
    {"units": [64, 128, 128]},
    {"units": [128, 128, 128]},

    {"units": [128, 32, 16]}
]



# Classification LSTM Stack 3 with Dropout
# C_LSTM_S3_DO
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_LSTM_S3_DO: List[IKerasModelConfig] = [
    {"units": [16, 16, 16], "dropout_rates": [0, 0, 0]},

    {"units": [32, 16, 16], "dropout_rates": [0, 0, 0]},
    {"units": [16, 32, 16], "dropout_rates": [0, 0, 0]},
    {"units": [16, 16, 32], "dropout_rates": [0, 0, 0]},
    {"units": [32, 32, 16], "dropout_rates": [0, 0, 0]},
    {"units": [16, 32, 32], "dropout_rates": [0, 0, 0]},
    {"units": [32, 32, 32], "dropout_rates": [0, 0, 0]},

    {"units": [64, 32, 32], "dropout_rates": [0, 0, 0]},
    {"units": [32, 64, 32], "dropout_rates": [0, 0, 0]},
    {"units": [32, 32, 64], "dropout_rates": [0, 0, 0]},
    {"units": [64, 64, 32], "dropout_rates": [0, 0, 0]},
    {"units": [32, 64, 64], "dropout_rates": [0, 0, 0]},
    {"units": [64, 64, 64], "dropout_rates": [0, 0, 0]},

    {"units": [64, 32, 16], "dropout_rates": [0, 0, 0]},

    {"units": [128, 64, 64], "dropout_rates": [0, 0, 0]},
    {"units": [64, 128, 64], "dropout_rates": [0, 0, 0]},
    {"units": [64, 64, 128], "dropout_rates": [0, 0, 0]},
    {"units": [128, 128, 64], "dropout_rates": [0, 0, 0]},
    {"units": [64, 128, 128], "dropout_rates": [0, 0, 0]},
    {"units": [128, 128, 128], "dropout_rates": [0, 0, 0]},

    {"units": [128, 32, 16], "dropout_rates": [0, 0, 0]}
]




# Classification LSTM Stack 4
# C_LSTM_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
C_LSTM_S4: List[IKerasModelConfig] = [
    {"units": [16, 16, 16, 16]},

    {"units": [32, 16, 16, 16]},
    {"units": [16, 32, 16, 16]},
    {"units": [16, 16, 32, 16]},
    {"units": [16, 16, 16, 32]},
    {"units": [32, 32, 16, 16]},
    {"units": [16, 16, 32, 32]},
    {"units": [32, 32, 32, 16]},
    {"units": [16, 32, 32, 32]},
    {"units": [32, 32, 32, 32]},

    {"units": [64, 32, 32, 32]},
    {"units": [32, 64, 32, 32]},
    {"units": [32, 32, 64, 32]},
    {"units": [32, 32, 32, 64]},
    {"units": [64, 64, 32, 32]},
    {"units": [32, 32, 64, 64]},
    {"units": [64, 64, 64, 32]},
    {"units": [32, 64, 64, 64]},
    {"units": [64, 64, 64, 64]},

    {"units": [64, 32, 16, 16]},

    {"units": [128, 64, 64, 64]},
    {"units": [64, 128, 64, 64]},
    {"units": [64, 64, 128, 64]},
    {"units": [64, 64, 64, 128]},
    {"units": [128, 128, 64, 64]},
    {"units": [64, 64, 128, 128]},
    {"units": [128, 128, 128, 64]},
    {"units": [64, 128, 128, 128]},
    {"units": [128, 128, 128, 128]},

    {"units": [128, 64, 32, 16]}
]






# Classification LSTM Stack 4 with Dropout
# C_LSTM_S4_DO
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_LSTM_S4_DO: List[IKerasModelConfig] = [
    {"units": [16, 16, 16, 16], "dropout_rates": [0, 0, 0, 0]},

    {"units": [32, 16, 16, 16], "dropout_rates": [0, 0, 0, 0]},
    {"units": [16, 32, 16, 16], "dropout_rates": [0, 0, 0, 0]},
    {"units": [16, 16, 32, 16], "dropout_rates": [0, 0, 0, 0]},
    {"units": [16, 16, 16, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 32, 16, 16], "dropout_rates": [0, 0, 0, 0]},
    {"units": [16, 16, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 32, 32, 16], "dropout_rates": [0, 0, 0, 0]},
    {"units": [16, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},

    {"units": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 64, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 32, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 32, 32, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 32, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [32, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},

    {"units": [64, 32, 16, 16], "dropout_rates": [0, 0, 0, 0]},

    {"units": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 128, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 128, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 64, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 128, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 128, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},

    {"units": [128, 64, 32, 16], "dropout_rates": [0, 0, 0, 0]}
]








# Network Variations
LSTM: ILSTM = {
    "C_LSTM_S1": C_LSTM_S1,
    "C_LSTM_S2": C_LSTM_S2,
    "C_LSTM_S2_DO": C_LSTM_S2_DO,
    "C_LSTM_S3": C_LSTM_S3,
    "C_LSTM_S4": C_LSTM_S4,
    "C_LSTM_S4_DO": C_LSTM_S4_DO
}