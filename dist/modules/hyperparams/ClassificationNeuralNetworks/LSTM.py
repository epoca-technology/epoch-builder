from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################





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
# 1 units: LSTM_1
C_LSTM_S1: List[IKerasModelConfig] = [
    {"units": [32]},
    {"units": [64]},
    {"units": [128]},
    {"units": [256]},
    {"units": [512]}
]




# Classification LSTM Stack 2
# C_LSTM_S2
# 2 units: LSTM_1, LSTM_2
C_LSTM_S2: List[IKerasModelConfig] = [
    {"units": [32, 32]},

    {"units": [64, 32]},
    {"units": [64, 64]},

    {"units": [128, 32]},
    {"units": [128, 64]},
    {"units": [128, 128]},

    {"units": [256, 32]},
    {"units": [256, 64]},
    {"units": [256, 128]},
    {"units": [256, 256]},

    {"units": [512, 32]},
    {"units": [512, 64]},
    {"units": [512, 128]},
    {"units": [512, 256]},
    {"units": [512, 512]}
]





# Classification LSTM Stack 2 with Dropout
# C_LSTM_S2_DO
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_LSTM_S2_DO: List[IKerasModelConfig] = [
    {"units": [32, 32], "dropout_rates": [0, 0]},

    {"units": [64, 32], "dropout_rates": [0, 0]},
    {"units": [64, 64], "dropout_rates": [0, 0]},

    {"units": [128, 32], "dropout_rates": [0, 0]},
    {"units": [128, 64], "dropout_rates": [0, 0]},
    {"units": [128, 128], "dropout_rates": [0, 0]},

    {"units": [256, 32], "dropout_rates": [0, 0]},
    {"units": [256, 64], "dropout_rates": [0, 0]},
    {"units": [256, 128], "dropout_rates": [0, 0]},
    {"units": [256, 256], "dropout_rates": [0, 0]},

    {"units": [512, 32], "dropout_rates": [0, 0]},
    {"units": [512, 64], "dropout_rates": [0, 0]},
    {"units": [512, 128], "dropout_rates": [0, 0]},
    {"units": [512, 256], "dropout_rates": [0, 0]},
    {"units": [512, 512], "dropout_rates": [0, 0]}
]







# Classification LSTM Stack 3
# C_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
C_LSTM_S3: List[IKerasModelConfig] = [
    {"units": [32, 32, 32]},

    {"units": [64, 32, 32]},
    {"units": [64, 64, 64]},

    {"units": [128, 64, 32]},
    {"units": [128, 128, 64]},
    {"units": [128, 128, 128]},

    {"units": [256, 64, 32]},
    {"units": [256, 128, 64]},
    {"units": [256, 128, 128]},
    {"units": [256, 256, 256]},

    {"units": [512, 64, 32]},
    {"units": [512, 128, 64]},
    {"units": [512, 256, 128]},
    {"units": [512, 256, 256]},
    {"units": [512, 512, 512]}
]






# Classification LSTM Stack 3 with Dropout
# C_LSTM_S3_DO
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_LSTM_S3_DO: List[IKerasModelConfig] = [
    {"units": [32, 32, 32], "dropout_rates": [0, 0, 0]},

    {"units": [64, 32, 32], "dropout_rates": [0, 0, 0]},
    {"units": [64, 64, 64], "dropout_rates": [0, 0, 0]},

    {"units": [128, 64, 32], "dropout_rates": [0, 0, 0]},
    {"units": [128, 128, 64], "dropout_rates": [0, 0, 0]},
    {"units": [128, 128, 128], "dropout_rates": [0, 0, 0]},

    {"units": [256, 64, 32], "dropout_rates": [0, 0, 0]},
    {"units": [256, 128, 64], "dropout_rates": [0, 0, 0]},
    {"units": [256, 128, 128], "dropout_rates": [0, 0, 0]},
    {"units": [256, 256, 256], "dropout_rates": [0, 0, 0]},

    {"units": [512, 64, 32], "dropout_rates": [0, 0, 0]},
    {"units": [512, 128, 64], "dropout_rates": [0, 0, 0]},
    {"units": [512, 256, 128], "dropout_rates": [0, 0, 0]},
    {"units": [512, 256, 256], "dropout_rates": [0, 0, 0]},
    {"units": [512, 512, 512], "dropout_rates": [0, 0, 0]}
]







# Classification LSTM Stack 4
# C_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
C_LSTM_S4: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32]},

    {"units": [64, 32, 32, 32]},
    {"units": [64, 64, 64, 64]},

    {"units": [128, 64, 64, 32]},
    {"units": [128, 128, 64, 64]},
    {"units": [128, 128, 128, 128]},

    {"units": [256, 128, 64, 32]},
    {"units": [256, 128, 128, 64]},
    {"units": [256, 128, 128, 128]},
    {"units": [256, 256, 256, 256]},

    {"units": [512, 128, 64, 32]},
    {"units": [512, 256, 128, 64]},
    {"units": [512, 256, 256, 128]},
    {"units": [512, 256, 256, 256]},
    {"units": [512, 512, 512, 512]}
]






# Classification LSTM Stack 4 with Dropout
# C_LSTM_S4_DO
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_LSTM_S4_DO: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},

    {"units": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},

    {"units": [128, 64, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},

    {"units": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 128, 128, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0]},

    {"units": [512, 128, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [512, 256, 256, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [512, 256, 256, 256], "dropout_rates": [0, 0, 0, 0]},
    {"units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0]}
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