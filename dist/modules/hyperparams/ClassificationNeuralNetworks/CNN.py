from typing import List, TypedDict
from modules.types import IKerasModelConfig


##################################
## Convolutional Neural Network ##
##################################




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
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 activations:    Conv1D_1
C_CNN_S1: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "activations": [""]}
]




# Classification CNN Stack 1 with MaxPooling
# C_CNN_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 activations:    Conv1D_1
C_CNN_S1_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]}
]





# Classification CNN Stack 1 with MaxPooling and Dropout
# C_CNN_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 dropout_rates:  Dropout_1
C_CNN_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]}
]






# Classification CNN Stack 2
# C_CNN_S2
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
C_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [64, 64], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [128, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [128, 64], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [128, 128], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [256, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [256, 64], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [256, 128], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [256, 256], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [512, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [512, 64], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [512, 128], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [512, 256], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [512, 512], "kernel_sizes": [5, 3], "activations": ["", ""]}
]




# Classification CNN Stack 2 with Dropout
# C_CNN_S2_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 2 activations:    Conv1D_1, Conv1D_2
C_CNN_S2_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 64], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [128, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 64], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 128], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [256, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 64], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 128], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 256], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [512, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 64], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 128], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 256], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 512], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]}
]





# Classification CNN Stack 2 with MaxPooling
# C_CNN_S2_MP
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 activations:    Conv1D_1, Conv1D_2
C_CNN_S2_MP: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [64, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [128, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [128, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [128, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [256, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [256, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [256, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [256, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [512, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [512, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [512, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [512, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [512, 512], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]}
]





# Classification CNN Stack 2 with MaxPooling and Dropout
# C_CNN_S2_MP_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
C_CNN_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [64, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [128, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [128, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [256, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [256, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [512, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"filters": [512, 512], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]}
]







# Classification CNN Stack 3
# C_CNN_S3
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
C_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [128, 64, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [256, 64, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [512, 64, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]}
]



# Classification CNN Stack 3 with Dropout
# C_CNN_S3_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
C_CNN_S3_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 64, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 64, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]}
]




# Classification CNN Stack 3 with MaxPooling
# C_CNN_S3_MP
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
C_CNN_S3_MP: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [128, 64, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [256, 64, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [512, 64, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]}
]








# Classification CNN Stack 3 with MaxPooling and Dropout
# C_CNN_S3_MP_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
C_CNN_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 64, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 64, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 128, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [512, 512, 512], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]}
]






# Classification CNN Stack 4
# C_CNN_S4
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
C_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [512, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
]




# Classification CNN Stack 4 with Dropout
# C_CNN_S4_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
C_CNN_S4_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 64], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
]




# Classification CNN Stack 4 with MaxPooling
# C_CNN_S4_MP
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
C_CNN_S4_MP: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [512, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
]






# Classification CNN Stack 4 with MaxPooling and Dropout
# C_CNN_S4_MP_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_CNN_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [512, 512, 512, 512], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
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