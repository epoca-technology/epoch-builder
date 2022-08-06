from typing import List
from modules._types import IKerasModelConfig, IKerasHyperparamsNetworkVariations


#####################################################
## Long Short-Term Memory Recurrent Neural Network ##
#####################################################




# Classification LSTM Stack 2
# KC_LSTM_S2
# 2 units: LSTM_1, LSTM_2
KC_LSTM_S2: List[IKerasModelConfig] = [
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



# Classification LSTM Stack 3
# KC_LSTM_S3
# 3 units: LSTM_1, LSTM_2, LSTM_3
KC_LSTM_S3: List[IKerasModelConfig] = [
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



# Classification LSTM Stack 4
# KC_LSTM_S4
# 4 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4
KC_LSTM_S4: List[IKerasModelConfig] = [
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


# Classification LSTM Stack 5
# KC_LSTM_S5
# 5 units: LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
KC_LSTM_S5: List[IKerasModelConfig] = [
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
LSTM: IKerasHyperparamsNetworkVariations = {
    "KC_LSTM_S2": KC_LSTM_S2,
    "KC_LSTM_S3": KC_LSTM_S3,
    "KC_LSTM_S4": KC_LSTM_S4,
    "KC_LSTM_S5": KC_LSTM_S5
}