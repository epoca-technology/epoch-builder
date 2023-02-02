from typing import List
from modules._types import IKerasModelConfig, IRegressionTrainingConfigNetworks


###################################################################
## Bidirectional Long Short-Term Memory Recurrent Neural Network ##
###################################################################




# Regression BDLSTM Stack 2
# KR_BDLSTM_S2
# 2 units: BDLSTM_1, BDLSTM_2
KR_BDLSTM_S2: List[IKerasModelConfig] = [
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





# Regression BDLSTM Stack 3
# KR_BDLSTM_S3
# 3 units: BDLSTM_1, BDLSTM_2, BDLSTM_3
KR_BDLSTM_S3: List[IKerasModelConfig] = [
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







# Regression BDLSTM Stack 4
# KR_BDLSTM_S4
# 4 units: BDLSTM_1, BDLSTM_2, BDLSTM_3, BDLSTM_4
KR_BDLSTM_S4: List[IKerasModelConfig] = [
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







# Regression BDLSTM Stack 5
# KR_BDLSTM_S5
# 5 units: BDLSTM_1, BDLSTM_2, BDLSTM_3, BDLSTM_4, BDLSTM_5
KR_BDLSTM_S5: List[IKerasModelConfig] = [
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
BDLSTM: IRegressionTrainingConfigNetworks = {
    "KR_BDLSTM_S2": KR_BDLSTM_S2,
    "KR_BDLSTM_S3": KR_BDLSTM_S3,
    "KR_BDLSTM_S4": KR_BDLSTM_S4,
    "KR_BDLSTM_S5": KR_BDLSTM_S5
}