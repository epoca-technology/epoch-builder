from typing import List, TypedDict
from modules.types import IKerasModelConfig


#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################





# Network Type
class ILSTM(TypedDict):
    R_LSTM_S1: List[IKerasModelConfig]
    R_LSTM_S2: List[IKerasModelConfig]
    R_LSTM_S3: List[IKerasModelConfig]
    R_LSTM_S4: List[IKerasModelConfig]
    R_LSTM_S5: List[IKerasModelConfig]



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

    {"units": [256, 64]},
    {"units": [256, 128]},
    {"units": [256, 256]},

    {"units": [512, 128]},
    {"units": [512, 256]},
    {"units": [512, 512]}
]



# Regression LSTM Stack 3
# R_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
R_LSTM_S3: List[IKerasModelConfig] = [
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

    {"units": [512, 128, 64]},
    {"units": [512, 256, 128]},
    {"units": [512, 256, 256]},
    {"units": [512, 512, 512]}
]



# Regression LSTM Stack 4
# R_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
R_LSTM_S4: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32]},

    {"units": [64, 32, 32, 32]},
    {"units": [64, 64, 64, 64]},

    {"units": [128, 64, 64, 32]},
    {"units": [128, 64, 64, 64]},
    {"units": [128, 128, 128, 128]},

    {"units": [256, 128, 64, 32]},
    {"units": [256, 128, 128, 64]},
    {"units": [256, 128, 128, 128]},
    {"units": [256, 256, 256, 256]},

    {"units": [512, 256, 128, 64]},
    {"units": [512, 256, 256, 128]},
    {"units": [512, 256, 256, 256]},
    {"units": [512, 512, 512, 512]}
]



# Regression LSTM Stack 5
# R_LSTM_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
R_LSTM_S5: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32, 32]},

    {"units": [64, 32, 32, 32, 32]},
    {"units": [64, 64, 64, 64, 64]},

    {"units": [128, 64, 64, 32, 32]},
    {"units": [128, 64, 64, 64, 64]},
    {"units": [128, 128, 128, 128, 128]},

    {"units": [256, 128, 64, 32, 32]},
    {"units": [256, 128, 128, 64, 64]},
    {"units": [256, 128, 128, 128, 128]},
    {"units": [256, 256, 256, 256, 256]},

    {"units": [512, 256, 128, 64, 32]},
    {"units": [512, 256, 256, 128, 128]},
    {"units": [512, 256, 256, 256, 256]},
    {"units": [512, 512, 512, 512, 512]}
]




# Network Variations
LSTM: ILSTM = {
    "R_LSTM_S1": R_LSTM_S1,
    "R_LSTM_S2": R_LSTM_S2,
    "R_LSTM_S3": R_LSTM_S3,
    "R_LSTM_S4": R_LSTM_S4,
    "R_LSTM_S5": R_LSTM_S5
}