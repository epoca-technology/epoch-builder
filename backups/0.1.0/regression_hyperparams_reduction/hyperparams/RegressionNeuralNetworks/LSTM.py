from typing import List, TypedDict
from modules.types import IKerasModelConfig


#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################





# Network Type
class ILSTM(TypedDict):
    R_LSTM_S1: List[IKerasModelConfig]
    R_LSTM_S2: List[IKerasModelConfig]
    R_LSTM_S2_DO: List[IKerasModelConfig]
    R_LSTM_S3: List[IKerasModelConfig]
    R_LSTM_S3_DO: List[IKerasModelConfig]
    R_LSTM_S4: List[IKerasModelConfig]
    R_LSTM_S4_DO: List[IKerasModelConfig]






# Regression LSTM Stack 1
# R_LSTM_S1
# 1 units: LSTM_1
R_LSTM_S1: List[IKerasModelConfig] = [
    {"units": [32]},
    {"units": [64]},
    {"units": [128]},
    {"units": [256]},
    {"units": [512]}
]




# Regression LSTM Stack 2
# R_LSTM_S2
# 2 units: LSTM_1, LSTM_2
R_LSTM_S2: List[IKerasModelConfig] = [
    {"units": [32, 32]},

    {"units": [64, 32]},
    {"units": [64, 64]},

    {"units": [128, 64]},
    {"units": [128, 128]},

    {"units": [256, 128]},
    {"units": [256, 256]},

    {"units": [512, 256]},
    {"units": [512, 512]}
]





# Regression LSTM Stack 2 with Dropout
# R_LSTM_S2_DO
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
R_LSTM_S2_DO: List[IKerasModelConfig] = [
    {"units": [32, 32], "dropout_rates": [0, 0]},

    {"units": [64, 32], "dropout_rates": [0, 0]},
    {"units": [64, 64], "dropout_rates": [0, 0]},

    {"units": [128, 64], "dropout_rates": [0, 0]},
    {"units": [128, 128], "dropout_rates": [0, 0]},

    {"units": [256, 128], "dropout_rates": [0, 0]},
    {"units": [256, 256], "dropout_rates": [0, 0]},

    {"units": [512, 256], "dropout_rates": [0, 0]},
    {"units": [512, 512], "dropout_rates": [0, 0]}
]







# Regression LSTM Stack 3
# R_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
R_LSTM_S3: List[IKerasModelConfig] = [
    {"units": [32, 32, 32]},

    {"units": [64, 32, 32]},
    {"units": [64, 64, 64]},

    {"units": [128, 128, 64]},
    {"units": [128, 128, 128]},

    {"units": [256, 128, 128]},
    {"units": [256, 256, 256]},

    {"units": [512, 256, 256]},
    {"units": [512, 512, 512]}
]






# Regression LSTM Stack 3 with Dropout
# R_LSTM_S3_DO
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
R_LSTM_S3_DO: List[IKerasModelConfig] = [
    {"units": [32, 32, 32], "dropout_rates": [0, 0, 0]},

    {"units": [64, 32, 32], "dropout_rates": [0, 0, 0]},
    {"units": [64, 64, 64], "dropout_rates": [0, 0, 0]},

    {"units": [128, 128, 64], "dropout_rates": [0, 0, 0]},
    {"units": [128, 128, 128], "dropout_rates": [0, 0, 0]},

    {"units": [256, 128, 128], "dropout_rates": [0, 0, 0]},
    {"units": [256, 256, 256], "dropout_rates": [0, 0, 0]},

    {"units": [512, 256, 256], "dropout_rates": [0, 0, 0]},
    {"units": [512, 512, 512], "dropout_rates": [0, 0, 0]}
]







# Regression LSTM Stack 4
# R_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
R_LSTM_S4: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32]},

    {"units": [64, 32, 32, 32]},
    {"units": [64, 64, 64, 64]},

    {"units": [128, 128, 64, 64]},
    {"units": [128, 128, 128, 128]},

    {"units": [256, 128, 128, 128]},
    {"units": [256, 256, 256, 256]},

    {"units": [512, 256, 256, 256]},
    {"units": [512, 512, 512, 512]}
]






# Regression LSTM Stack 4 with Dropout
# R_LSTM_S4_DO
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
R_LSTM_S4_DO: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},

    {"units": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},

    {"units": [128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},

    {"units": [256, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0]},

    {"units": [512, 256, 256, 256], "dropout_rates": [0, 0, 0, 0]},
    {"units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0]}
]








# Network Variations
LSTM: ILSTM = {
    "R_LSTM_S1": R_LSTM_S1,
    "R_LSTM_S2": R_LSTM_S2,
    "R_LSTM_S2_DO": R_LSTM_S2_DO,
    "R_LSTM_S3": R_LSTM_S3,
    "R_LSTM_S4": R_LSTM_S4,
    "R_LSTM_S4_DO": R_LSTM_S4_DO
}