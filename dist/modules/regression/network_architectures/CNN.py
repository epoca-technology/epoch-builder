from typing import List
from modules._types import IKerasModelConfig, IRegressionTrainingConfigNetworks


########################################
## Convolutional Dense Neural Network ##
########################################






# Regression CNN Stack 2
# KR_CNN_S2
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
KR_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [64, 64], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [128, 128], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [256, 256], "kernel_sizes": [5, 3], "activations": ["", ""]},
    {"filters": [512, 128], "kernel_sizes": [5, 3], "activations": ["", ""]}
]






# Regression CNN Stack 2 with MaxPooling
# KR_CNN_MP_S2
# 2 filters:        Conv1D_1, Conv1D_2
# 2 kernel_sizes:   Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
KR_CNN_MP_S2: List[IKerasModelConfig] = [
    {"filters": [32, 32], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [64, 64], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [128, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [256, 256], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]},
    {"filters": [512, 128], "kernel_sizes": [5, 3], "pool_sizes": [2, 2], "activations": ["", ""]}
]










# Regression CNN Stack 3
# KR_CNN_S3
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
KR_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]},
    {"filters": [512, 128, 32], "kernel_sizes": [5, 3, 3], "activations": ["", "", ""]}
]









# Regression CNN Stack 3 with MaxPooling
# KR_CNN_MP_S3
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
KR_CNN_MP_S3: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [256, 256, 256], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]},
    {"filters": [512, 128, 32], "kernel_sizes": [5, 3, 3], "pool_sizes": [2, 2, 2], "activations": ["", "", ""]}
]









# Regression CNN Stack 4
# KR_CNN_S4
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
KR_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "activations": ["", "", "", ""]}
]








# Regression CNN Stack 4 with MaxPooling
# KR_CNN_MP_S4
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
KR_CNN_MP_S4: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]},
    {"filters": [512, 256, 128, 64], "kernel_sizes": [5, 3, 3, 3], "pool_sizes": [2, 2, 2, 2], "activations": ["", "", "", ""]}
]








# Regression CNN Stack 5
# KR_CNN_S5
# 5 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
KR_CNN_S5: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3, 3], "activations": ["", "", "", "", ""]},
    {"filters": [64, 64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3, 3], "activations": ["", "", "", "", ""]},
    {"filters": [128, 128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3, 3], "activations": ["", "", "", "", ""]},
    {"filters": [256, 256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3, 3], "activations": ["", "", "", "", ""]},
    {"filters": [512, 256, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3, 3], "activations": ["", "", "", "", ""]}
]







# Regression CNN Stack 5 with MaxPooling
# KR_CNN_MP_S5
# 5 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 kernel_sizes:   Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4, Conv1D_5
# 5 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4, MaxPooling1D_5
KR_CNN_MP_S5: List[IKerasModelConfig] = [
    {"filters": [32, 32, 32, 32, 32], "kernel_sizes": [5, 3, 3, 3, 3], "pool_sizes": [2, 2, 2, 2, 2], "activations": ["", "", "", "", ""]},
    {"filters": [64, 64, 64, 64, 64], "kernel_sizes": [5, 3, 3, 3, 3], "pool_sizes": [2, 2, 2, 2, 2], "activations": ["", "", "", "", ""]},
    {"filters": [128, 128, 128, 128, 128], "kernel_sizes": [5, 3, 3, 3, 3], "pool_sizes": [2, 2, 2, 2, 2], "activations": ["", "", "", "", ""]},
    {"filters": [256, 256, 256, 256, 256], "kernel_sizes": [5, 3, 3, 3, 3], "pool_sizes": [2, 2, 2, 2, 2], "activations": ["", "", "", "", ""]},
    {"filters": [512, 256, 128, 64, 32], "kernel_sizes": [5, 3, 3, 3, 3], "pool_sizes": [2, 2, 2, 2, 2], "activations": ["", "", "", "", ""]}
]







# Network Variations
CNN: IRegressionTrainingConfigNetworks = {
    "KR_CNN_S2": KR_CNN_S2,
    "KR_CNN_MP_S2": KR_CNN_MP_S2,
    "KR_CNN_S3": KR_CNN_S3,
    "KR_CNN_MP_S3": KR_CNN_MP_S3,
    "KR_CNN_S4": KR_CNN_S4,
    "KR_CNN_MP_S4": KR_CNN_MP_S4,
    "KR_CNN_S5": KR_CNN_S5,
    "KR_CNN_MP_S5": KR_CNN_MP_S5
}