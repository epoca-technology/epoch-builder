from typing import List
from modules._types import IKerasModelConfig, IKerasHyperparamsNetworkVariations


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




# Regression LSTM Stack 2 with Dropout
# KR_LSTM_DO_S2
# 2 units: LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
KR_LSTM_DO_S2: List[IKerasModelConfig] = [
    {"units": [32, 32], "dropout_rates": [0, 0]},

    {"units": [64, 32], "dropout_rates": [0, 0]},
    {"units": [64, 64], "dropout_rates": [0, 0]},

    {"units": [128, 64], "dropout_rates": [0, 0]},
    {"units": [128, 128], "dropout_rates": [0, 0]},

    {"units": [256, 64], "dropout_rates": [0, 0]},
    {"units": [256, 128], "dropout_rates": [0, 0]},
    {"units": [256, 256], "dropout_rates": [0, 0]},

    {"units": [512, 128], "dropout_rates": [0, 0]}
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




# Regression LSTM Stack 3 with Dropout
# KR_LSTM_DO_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
KR_LSTM_DO_S3: List[IKerasModelConfig] = [
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

    {"units": [512, 128, 64], "dropout_rates": [0, 0, 0]}
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





# Regression LSTM Stack 4 with Dropout
# KR_LSTM_DO_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
KR_LSTM_DO_S4: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},

    {"units": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},

    {"units": [128, 64, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},

    {"units": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 128, 128, 64], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 128, 128, 128], "dropout_rates": [0, 0, 0, 0]},
    {"units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0]},

    {"units": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0]}
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




# Regression LSTM Stack 5 with Dropout
# KR_LSTM_DO_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
KR_LSTM_DO_S5: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0, 0]},

    {"units": [64, 32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0, 0]},
    {"units": [64, 64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0, 0]},

    {"units": [128, 64, 64, 32, 32], "dropout_rates": [0, 0, 0, 0, 0]},
    {"units": [128, 64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0, 0]},
    {"units": [128, 128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0, 0]},

    {"units": [256, 128, 64, 32, 32], "dropout_rates": [0, 0, 0, 0, 0]},
    {"units": [256, 128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0, 0]},
    {"units": [256, 128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0, 0]},
    {"units": [256, 256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0, 0]},

    {"units": [512, 256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0, 0]}
]





# Network Variations
LSTM: IKerasHyperparamsNetworkVariations = {
    "KR_LSTM_S2": KR_LSTM_S2,
    "KR_LSTM_DO_S2": KR_LSTM_DO_S2,
    "KR_LSTM_S3": KR_LSTM_S3,
    "KR_LSTM_DO_S3": KR_LSTM_DO_S3,
    "KR_LSTM_S4": KR_LSTM_S4,
    "KR_LSTM_DO_S4": KR_LSTM_DO_S4,
    "KR_LSTM_S5": KR_LSTM_S5,
    "KR_LSTM_DO_S5": KR_LSTM_DO_S5
}