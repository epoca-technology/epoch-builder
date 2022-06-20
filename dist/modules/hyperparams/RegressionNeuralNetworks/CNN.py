from typing import List, TypedDict
from modules.types import IKerasModelConfig


##################################
## Convolutional Neural Network ##
##################################




# Network Type
class ICNN(TypedDict):
    R_CNN_S1: List[IKerasModelConfig]
    R_CNN_S1_MP: List[IKerasModelConfig]
    R_CNN_S1_MP_DO: List[IKerasModelConfig]
    R_CNN_S2: List[IKerasModelConfig]
    R_CNN_S2_DO: List[IKerasModelConfig]
    R_CNN_S2_MP: List[IKerasModelConfig]
    R_CNN_S2_MP_DO: List[IKerasModelConfig]
    R_CNN_S3: List[IKerasModelConfig]
    R_CNN_S3_DO: List[IKerasModelConfig]
    R_CNN_S3_MP: List[IKerasModelConfig]
    R_CNN_S3_MP_DO: List[IKerasModelConfig]
    R_CNN_S4: List[IKerasModelConfig]
    R_CNN_S4_DO: List[IKerasModelConfig]
    R_CNN_S4_MP: List[IKerasModelConfig]
    R_CNN_S4_MP_DO: List[IKerasModelConfig]







# Regression CNN Stack 1
# R_CNN_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 activations:    Conv1D_1
R_CNN_S1: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "activations": [""]}
]




# Regression CNN Stack 1 with MaxPooling
# R_CNN_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 activations:    Conv1D_1
R_CNN_S1_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "activations": [""]}
]





# Regression CNN Stack 1 with MaxPooling and Dropout
# R_CNN_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 dropout_rates:  Dropout_1
R_CNN_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "dropout_rates": [0], "activations": [""]}
]






# Regression CNN Stack 2
# R_CNN_S2
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
R_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [128, 64], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [256, 128], "kernel_sizes": [5, 3], "activations": ["", ""]},

    {"filters": [512, 256], "kernel_sizes": [5, 3], "activations": ["", ""]}
]




# Regression CNN Stack 2 with Dropout
# R_CNN_S2_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 2 activations:    Conv1D_1, Conv1D_2
R_CNN_S2_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [128, 64], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [256, 128], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [512, 256], "kernel_sizes": [5, 3], "dropout_rates": [0, 0], "activations": ["", ""]}
]





# Regression CNN Stack 2 with MaxPooling
# R_CNN_S2_MP
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 activations:    Conv1D_1, Conv1D_2
R_CNN_S2_MP: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [128, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [256, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},

    {"filters": [512, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]}
]





# Regression CNN Stack 2 with MaxPooling and Dropout
# R_CNN_S2_MP_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
R_CNN_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [128, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [256, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"filters": [512, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "dropout_rates": [0, 0], "activations": ["", ""]}
]







# Regression CNN Stack 3
# R_CNN_S3
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
R_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]}
]



# Regression CNN Stack 3 with Dropout
# R_CNN_S3_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
R_CNN_S3_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]}
]




# Regression CNN Stack 3 with MaxPooling
# R_CNN_S3_MP
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
R_CNN_S3_MP: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]}
]








# Regression CNN Stack 3 with MaxPooling and Dropout
# R_CNN_S3_MP_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
R_CNN_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [256, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [512, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]}
]






# Regression CNN Stack 4
# R_CNN_S4
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
R_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]}
]




# Regression CNN Stack 4 with Dropout
# R_CNN_S4_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
R_CNN_S4_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]




# Regression CNN Stack 4 with MaxPooling
# R_CNN_S4_MP
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
R_CNN_S4_MP: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]}
]






# Regression CNN Stack 4 with MaxPooling and Dropout
# R_CNN_S4_MP_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
R_CNN_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [256, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]







# Network Variations
CNN: ICNN = {
    "R_CNN_S1": R_CNN_S1,
    "R_CNN_S1_MP": R_CNN_S1_MP,
    "R_CNN_S1_MP_DO": R_CNN_S1_MP_DO,
    "R_CNN_S2": R_CNN_S2,
    "R_CNN_S2_DO": R_CNN_S2_DO,
    "R_CNN_S2_MP": R_CNN_S2_MP,
    "R_CNN_S2_MP_DO": R_CNN_S2_MP_DO,
    "R_CNN_S3": R_CNN_S3,
    "R_CNN_S3_DO": R_CNN_S3_DO,
    "R_CNN_S3_MP": R_CNN_S3_MP,
    "R_CNN_S3_MP_DO": R_CNN_S3_MP_DO,
    "R_CNN_S4": R_CNN_S4,
    "R_CNN_S4_DO": R_CNN_S4_DO,
    "R_CNN_S4_MP": R_CNN_S4_MP,
    "R_CNN_S4_MP_DO": R_CNN_S4_MP_DO
}