from typing import List
from modules._types import IKerasModelConfig, IRegressionTrainingConfigNetworks


#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################




# Regression LSTM Stack 2
# KR_LSTM_S2
# 2 units: LSTM_1, LSTM_2
KR_LSTM_S2: List[IKerasModelConfig] = [
    {"units": [32, 32]},

    {"units": [64, 32]},
    {"units": [64, 64]},

    {"units": [128, 64]},
    {"units": [128, 128]},

    {"units": [256, 64]},
    {"units": [256, 128]},
    {"units": [256, 256]},

    {"units": [512, 128]}
]





# Regression LSTM Stack 3
# KR_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
KR_LSTM_S3: List[IKerasModelConfig] = [
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

    {"units": [512, 128, 64]}
]







# Regression LSTM Stack 4
# KR_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
KR_LSTM_S4: List[IKerasModelConfig] = [
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

    {"units": [512, 256, 128, 64]}
]







# Regression LSTM Stack 5
# KR_LSTM_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
KR_LSTM_S5: List[IKerasModelConfig] = [
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

    {"units": [512, 256, 128, 64, 32]}
]







# Network Variations
LSTM: IRegressionTrainingConfigNetworks = {
    "KR_LSTM_S2": KR_LSTM_S2,
    "KR_LSTM_S3": KR_LSTM_S3,
    "KR_LSTM_S4": KR_LSTM_S4,
    "KR_LSTM_S5": KR_LSTM_S5
}