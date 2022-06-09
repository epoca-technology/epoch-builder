from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


## Convolutional Neural Network ##


# Network Type
class ICNN(TypedDict):
    C_CNN_S1: List[IKerasModelConfig]
    C_CNN_S1_MP: List[IKerasModelConfig]
    C_CNN_S1_MP_DO: List[IKerasModelConfig]
    C_CNN_S2: List[IKerasModelConfig]
    C_CNN_S2_DO: List[IKerasModelConfig]
    C_CNN_S2_MP: List[IKerasModelConfig]
    C_CNN_S2_MP_DO: List[IKerasModelConfig]
    C_CNN_S3: List[IKerasModelConfig]
    C_CNN_S3_DO: List[IKerasModelConfig]
    C_CNN_S3_MP: List[IKerasModelConfig]
    C_CNN_S3_MP_DO: List[IKerasModelConfig]
    C_CNN_S4: List[IKerasModelConfig]
    C_CNN_S4_DO: List[IKerasModelConfig]
    C_CNN_S4_MP: List[IKerasModelConfig]
    C_CNN_S4_MP_DO: List[IKerasModelConfig]





# Classification CNN Stack 1
# C_CNN_S1
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
C_CNN_S1: List[IKerasModelConfig] = [
    {"filters": [8], "activations": [""]},
    {"filters": [16], "activations": [""]},
    {"filters": [32], "activations": [""]},
    {"filters": [64], "activations": [""]},
    {"filters": [128], "activations": [""]},
    {"filters": [256], "activations": [""]},
    {"filters": [512], "activations": [""]}
]



# Classification CNN Stack 1 with MaxPooling
# C_CNN_S1_MP
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
# 1 pool_sizes:  MaxPooling1D_1
C_CNN_S1_MP: List[IKerasModelConfig] = [
    {"filters": [8], "pool_sizes": [0], "activations": [""]},
    {"filters": [16], "pool_sizes": [0], "activations": [""]},
    {"filters": [32], "pool_sizes": [0], "activations": [""]},
    {"filters": [64], "pool_sizes": [0], "activations": [""]},
    {"filters": [128], "pool_sizes": [0], "activations": [""]},
    {"filters": [256], "pool_sizes": [0], "activations": [""]},
    {"filters": [512], "pool_sizes": [0], "activations": [""]}
]




# Classification CNN Stack 1 with MaxPooling and Dropout
# C_CNN_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 dropout_rates:  Dropout_1
C_CNN_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [8], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [16], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [32], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [64], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [128], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [256], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [512], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]}
]





# Classification CNN Stack 2
# C_CNN_S2
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
C_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [8, 8], "activations": ["", ""]},

    {"filters": [16, 8], "activations": ["", ""]},
    {"filters": [8, 16], "activations": ["", ""]},
    {"filters": [16, 16], "activations": ["", ""]},

    {"filters": [32, 16], "activations": ["", ""]},
    {"filters": [16, 32], "activations": ["", ""]},
    {"filters": [32, 32], "activations": ["", ""]},

    {"filters": [64, 8], "activations": ["", ""]},
    {"filters": [8, 64], "activations": ["", ""]},
    {"filters": [64, 16], "activations": ["", ""]},
    {"filters": [16, 64], "activations": ["", ""]},

    {"filters": [64, 32], "activations": ["", ""]},
    {"filters": [32, 64], "activations": ["", ""]},
    {"filters": [64, 64], "activations": ["", ""]},

    {"filters": [128, 8], "activations": ["", ""]},
    {"filters": [8, 128], "activations": ["", ""]},
    {"filters": [128, 16], "activations": ["", ""]},
    {"filters": [16, 128], "activations": ["", ""]},
    {"filters": [128, 32], "activations": ["", ""]},
    {"filters": [32, 128], "activations": ["", ""]},

    {"filters": [128, 64], "activations": ["", ""]},
    {"filters": [64, 128], "activations": ["", ""]},
    {"filters": [128, 128], "activations": ["", ""]},

    {"filters": [256, 8], "activations": ["", ""]},
    {"filters": [8, 256], "activations": ["", ""]},
    {"filters": [256, 16], "activations": ["", ""]},
    {"filters": [16, 256], "activations": ["", ""]},
    {"filters": [256, 32], "activations": ["", ""]},
    {"filters": [32, 256], "activations": ["", ""]},
    {"filters": [256, 64], "activations": ["", ""]},
    {"filters": [64, 256], "activations": ["", ""]},

    {"filters": [256, 128], "activations": ["", ""]},
    {"filters": [128, 256], "activations": ["", ""]},
    {"filters": [256, 256], "activations": ["", ""]},

    {"filters": [512, 8], "activations": ["", ""]},
    {"filters": [8, 512], "activations": ["", ""]},
    {"filters": [512, 16], "activations": ["", ""]},
    {"filters": [16, 512], "activations": ["", ""]},
    {"filters": [512, 32], "activations": ["", ""]},
    {"filters": [32, 512], "activations": ["", ""]},
    {"filters": [512, 64], "activations": ["", ""]},
    {"filters": [64, 512], "activations": ["", ""]},
    {"filters": [512, 128], "activations": ["", ""]},
    {"filters": [128, 512], "activations": ["", ""]},

    {"filters": [512, 256], "activations": ["", ""]},
    {"filters": [256, 512], "activations": ["", ""]},
    {"filters": [512, 512], "activations": ["", ""]}
]



# Classification CNN Stack 2 with Dropout
# C_CNN_S2_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_CNN_S2_DO: List[IKerasModelConfig] = [
    {"filters": [8, 8], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [16, 8], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [8, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [16, 16], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [32, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [16, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [32, 32], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [64, 8], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [8, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [16, 64], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [32, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 64], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [128, 8], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [8, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [16, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [32, 128], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [128, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 128], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [256, 8], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [8, 256], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [16, 256], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [32, 256], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 256], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [256, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 256], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 256], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [512, 8], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [8, 512], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [16, 512], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [32, 512], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 512], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 512], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [512, 256], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 512], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 512], "dropout_rates": [0, 0], "activations": ["", ""]}
]




# Classification CNN Stack 2 with MaxPooling
# C_CNN_S2_MP
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
# 2 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2
C_CNN_S2_MP: List[IKerasModelConfig] = [
    {"filters": [8, 8], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [16, 8], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 16], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 16], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [32, 16], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 32], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 32], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [64, 8], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 64], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 16], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 64], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 64], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 64], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [128, 8], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 128], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 16], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 128], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 32], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 128], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [128, 64], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 128], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 128], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [256, 8], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 256], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 16], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 256], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 32], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 256], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 64], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 256], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [256, 128], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 256], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 256], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [512, 8], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 512], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 16], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 512], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 32], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 512], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 64], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 512], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 128], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 512], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [512, 256], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 512], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 512], "pool_sizes": [0, 0], "activations": ["", ""]}
]





# Classification CNN Stack 2 with MaxPooling and Dropout
# C_CNN_S2_MP_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_CNN_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [8, 8], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [16, 8], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [32, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [64, 8], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [128, 8], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [128, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [256, 8], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [256, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [512, 8], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [8, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [16, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [512, 256], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [256, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [512, 512], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]}
]




# Classification CNN Stack 3
# C_CNN_S3
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
C_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8], "activations": ["", "", ""]},
    
    {"filters": [16, 8, 8], "activations": ["", "", ""]},
    {"filters": [8, 16, 8], "activations": ["", "", ""]},
    {"filters": [8, 8, 16], "activations": ["", "", ""]},
    {"filters": [16, 16, 8], "activations": ["", "", ""]},
    {"filters": [8, 16, 16], "activations": ["", "", ""]},
    {"filters": [16, 16, 16], "activations": ["", "", ""]},
    
    {"filters": [32, 16, 16], "activations": ["", "", ""]},
    {"filters": [16, 32, 16], "activations": ["", "", ""]},
    {"filters": [16, 16, 32], "activations": ["", "", ""]},
    {"filters": [32, 32, 16], "activations": ["", "", ""]},
    {"filters": [16, 32, 32], "activations": ["", "", ""]},
    {"filters": [32, 32, 32], "activations": ["", "", ""]},

    {"filters": [32, 16, 8], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "activations": ["", "", ""]},
    {"filters": [32, 64, 32], "activations": ["", "", ""]},
    {"filters": [32, 32, 64], "activations": ["", "", ""]},
    {"filters": [64, 64, 32], "activations": ["", "", ""]},
    {"filters": [32, 64, 64], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "activations": ["", "", ""]},

    {"filters": [64, 16, 8], "activations": ["", "", ""]},
    {"filters": [64, 32, 16], "activations": ["", "", ""]},
    {"filters": [64, 32, 8], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "activations": ["", "", ""]},
    {"filters": [64, 128, 64], "activations": ["", "", ""]},
    {"filters": [64, 64, 128], "activations": ["", "", ""]},
    {"filters": [128, 128, 64], "activations": ["", "", ""]},
    {"filters": [64, 128, 128], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "activations": ["", "", ""]},

    {"filters": [128, 16, 8], "activations": ["", "", ""]},
    {"filters": [128, 32, 16], "activations": ["", "", ""]},
    {"filters": [128, 64, 32], "activations": ["", "", ""]},
    {"filters": [128, 64, 16], "activations": ["", "", ""]},
    {"filters": [128, 64, 8], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "activations": ["", "", ""]},
    {"filters": [128, 256, 128], "activations": ["", "", ""]},
    {"filters": [128, 128, 256], "activations": ["", "", ""]},
    {"filters": [256, 256, 128], "activations": ["", "", ""]},
    {"filters": [128, 256, 256], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "activations": ["", "", ""]},

    {"filters": [256, 16, 8], "activations": ["", "", ""]},
    {"filters": [256, 32, 16], "activations": ["", "", ""]},
    {"filters": [256, 64, 32], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "activations": ["", "", ""]},
    {"filters": [256, 128, 32], "activations": ["", "", ""]},
    {"filters": [256, 128, 16], "activations": ["", "", ""]},
    {"filters": [256, 128, 8], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "activations": ["", "", ""]},
    {"filters": [256, 512, 256], "activations": ["", "", ""]},
    {"filters": [256, 256, 512], "activations": ["", "", ""]},
    {"filters": [512, 512, 256], "activations": ["", "", ""]},
    {"filters": [256, 512, 512], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "activations": ["", "", ""]},

    {"filters": [512, 16, 8], "activations": ["", "", ""]},
    {"filters": [512, 32, 16], "activations": ["", "", ""]},
    {"filters": [512, 64, 32], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "activations": ["", "", ""]},
    {"filters": [512, 256, 64], "activations": ["", "", ""]},
    {"filters": [512, 256, 32], "activations": ["", "", ""]},
    {"filters": [512, 256, 16], "activations": ["", "", ""]},
    {"filters": [512, 256, 8], "activations": ["", "", ""]},
]





# Classification CNN Stack 3 with Dropout
# C_CNN_S3_DO
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_CNN_S3_DO: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [16, 8, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 8, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 16, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [32, 16, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [32, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 32, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 256, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 256, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 512, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 512], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 512, 512], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 16, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 8], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
]






# Classification CNN Stack 3 with MaxPooling
# C_CNN_S3_MP
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
C_CNN_S3_MP: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [16, 8, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 8, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 16, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [32, 16, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 32, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 32, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [32, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 64, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 64, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 32, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 32, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 128, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 128, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 32, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 256, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 256], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 256, 256], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 32, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 64, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 512, 256], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 512], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 256], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 512, 512], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 16, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 32, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 64, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 64], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 32], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 16], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 8], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
]





# Classification CNN Stack 3 with MaxPooling and Dropout
# C_CNN_S3_MP_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_CNN_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [16, 8, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 8, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [8, 16, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [32, 16, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 16, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [16, 32, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [32, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 64, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 64, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 32, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 128, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 128, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 256, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 256], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 256, 256], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 64, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 512, 256], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 512], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 256], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 512, 512], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 16, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 64, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 8], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
]






# Classification CNN Stack 4
# C_CNN_S4
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
C_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8, 8], "activations": ["", "", "", ""]},

    {"filters": [16, 8, 8, 8], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 8, 8], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 8], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 8, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 8, 8], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 8], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 16], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 16], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 32], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 8, 8], "activations": ["", "", "", ""]},
    {"filters": [32, 16, 16, 8], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 8], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 8], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 32], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 16, 8], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 32, 16], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 16], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 16], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 8], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 8], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 16], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 32], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 32], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 32], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 16], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 8], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 8], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 16], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 32], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 64], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 64], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 32], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 16], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 8], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 512], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 512], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 256], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 512, 512], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 8], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 16], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 32], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 128], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 64], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 32], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 16], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 8], "activations": ["", "", "", ""]}
]




# Classification CNN Stack 4 with Dropout
# C_CNN_S4_DO
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_CNN_S4_DO: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [16, 8, 8, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 8, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 8, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 8, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 8, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 16, 16, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 16, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 512], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 8], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]





# Classification CNN Stack 4 with MaxPooling
# C_CNN_S4_MP
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:  MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
C_CNN_S4_MP: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [16, 8, 8, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 8, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 8, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 8, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 8, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 16, 16, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 16, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 16, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 32, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 256, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 256, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 256, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 512], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 512], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 256], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 512, 512], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 128], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 64], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 32], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 16], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 8], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]





# Classification CNN Stack 4 with MaxPooling and Dropout
# C_CNN_S4_MP_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_CNN_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [8, 8, 8, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [16, 8, 8, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 8, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 8, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 8, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 8, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [8, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 16, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 16, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [16, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 8, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 16, 16, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 16, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 16, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 32, 32, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 32, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 256, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 256, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 512], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 512, 512], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 256], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 256, 8], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]



# Network Variations
CNN: ICNN = {
    "C_CNN_S1": C_CNN_S1,
    "C_CNN_S1_MP": C_CNN_S1_MP,
    "C_CNN_S1_MP_DO": C_CNN_S1_MP_DO,
    "C_CNN_S2": C_CNN_S2,
    "C_CNN_S2_DO": C_CNN_S2_DO,
    "C_CNN_S2_MP": C_CNN_S2_MP,
    "C_CNN_S2_MP_DO": C_CNN_S2_MP_DO,
    "C_CNN_S3": C_CNN_S3,
    "C_CNN_S3_DO": C_CNN_S3_DO,
    "C_CNN_S3_MP": C_CNN_S3_MP,
    "C_CNN_S3_MP_DO": C_CNN_S3_MP_DO,
    "C_CNN_S4": C_CNN_S4,
    "C_CNN_S4_DO": C_CNN_S4_DO,
    "C_CNN_S4_MP": C_CNN_S4_MP,
    "C_CNN_S4_MP_DO": C_CNN_S4_MP_DO
}