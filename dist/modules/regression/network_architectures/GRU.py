from typing import List
from modules._types import IKerasModelConfig, IRegressionTrainingConfigNetworks


#########################################
## Gated Recurrent Unit Neural Network ##
#########################################




# Regression GRU Stack 2
# KR_GRU_S2
# 2 units: GRU_1, GRU_2
KR_GRU_S2: List[IKerasModelConfig] = [
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





# Regression GRU Stack 3
# KR_GRU_S3
# 3 units: GRU_1, GRU_2, GRU_3
KR_GRU_S3: List[IKerasModelConfig] = [
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







# Regression GRU Stack 4
# KR_GRU_S4
# 4 units: GRU_1, GRU_2, GRU_3, GRU_4
KR_GRU_S4: List[IKerasModelConfig] = [
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







# Regression GRU Stack 5
# KR_GRU_S5
# 5 units: GRU_1, GRU_2, GRU_3, GRU_4, GRU_5
KR_GRU_S5: List[IKerasModelConfig] = [
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
GRU: IRegressionTrainingConfigNetworks = {
    "KR_GRU_S2": KR_GRU_S2,
    "KR_GRU_S3": KR_GRU_S3,
    "KR_GRU_S4": KR_GRU_S4,
    "KR_GRU_S5": KR_GRU_S5
}