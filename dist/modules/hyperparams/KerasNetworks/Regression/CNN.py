from typing import List
from modules._types import IKerasModelConfig, IKerasHyperparamsNetworkVariations


##################################
## Convolutional Neural Network ##
##################################






# Regression CNN Stack 2
# KR_CNN_S2
# 1 filters:        Conv1D_1, Conv1D_2
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
KR_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128], "activations": ["", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128], "activations": ["", "", ""]}
]





# Regression CNN Stack 2 with MaxPooling
# KR_CNN_S2_MP
# 1 filters:        Conv1D_1, Conv1D_2
# 1 kernel_sizes:   Conv1D_1, Conv1D_2
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
KR_CNN_S2_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": ["", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": ["", "", ""]}
]












# Regression CNN Stack 3
# KR_CNN_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
KR_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128, 32], "activations": ["", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128, 32], "activations": ["", "", "", ""]}
]








# Regression CNN Stack 3 with MaxPooling
# KR_CNN_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
KR_CNN_S3_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": ["", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": ["", "", "", ""]}
]









# Regression CNN Stack 4
# KR_CNN_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
KR_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]}
]






# Regression CNN Stack 4 with MaxPooling
# KR_CNN_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
KR_CNN_S4_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]}
]





# Regression CNN Stack 5
# KR_CNN_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
KR_CNN_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]}
]




# Regression CNN Stack 5 with MaxPooling
# KR_CNN_S5_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
KR_CNN_S5_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]}
]





# Network Variations
CNN: IKerasHyperparamsNetworkVariations = {
    "KR_CNN_S2": KR_CNN_S2,
    "KR_CNN_S2_MP": KR_CNN_S2_MP,
    "KR_CNN_S3": KR_CNN_S3,
    "KR_CNN_S3_MP": KR_CNN_S3_MP,
    "KR_CNN_S4": KR_CNN_S4,
    "KR_CNN_S4_MP": KR_CNN_S4_MP,
    "KR_CNN_S5": KR_CNN_S5,
    "KR_CNN_S5_MP": KR_CNN_S5_MP
}