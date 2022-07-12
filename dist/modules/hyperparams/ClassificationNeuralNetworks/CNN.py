from typing import List, TypedDict
from modules.types import IKerasModelConfig


##################################
## Convolutional Neural Network ##
##################################




# Network Type
class ICNN(TypedDict):
    C_CNN_S1: List[IKerasModelConfig]
    C_CNN_S1_MP: List[IKerasModelConfig]
    C_CNN_S2: List[IKerasModelConfig]
    C_CNN_S2_MP: List[IKerasModelConfig]
    C_CNN_S3: List[IKerasModelConfig]
    C_CNN_S3_MP: List[IKerasModelConfig]
    C_CNN_S4: List[IKerasModelConfig]
    C_CNN_S4_MP: List[IKerasModelConfig]







# Classification CNN Stack 1
# C_CNN_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          Dense_1
# 2 activations:    Conv1D_1, Dense_1
C_CNN_S1: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32], "activations": ["", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512], "activations": ["", ""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64], "activations": ["", ""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512], "activations": ["", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [128], "activations": ["", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512], "activations": ["", ""]},

    {"filters": [256], "kernel_sizes": [3], "units": [256], "activations": ["", ""]},
    {"filters": [256], "kernel_sizes": [3], "units": [512], "activations": ["", ""]},

    {"filters": [512], "kernel_sizes": [3], "units": [512], "activations": ["", ""]}
]




# Classification CNN Stack 1 with MaxPooling
# C_CNN_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          Dense_1
# 2 activations:    Conv1D_1, Dense_1
C_CNN_S1_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32], "activations": ["", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "activations": ["", ""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64], "activations": ["", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "activations": ["", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128], "activations": ["", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "activations": ["", ""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256], "activations": ["", ""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "activations": ["", ""]},

    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "activations": ["", ""]}
]






# Classification CNN Stack 2
# C_CNN_S2
# 1 filters:        Conv1D_1, Conv1D_2
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
C_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [256], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [256], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [512], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]}
]





# Classification CNN Stack 2 with MaxPooling
# C_CNN_S2_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
C_CNN_S2_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]},

    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]}
]






# Classification CNN Stack 3
# C_CNN_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
C_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [256], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [512], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]}
]






# Classification CNN Stack 3 with MaxPooling
# C_CNN_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
C_CNN_S3_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]},

    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]}
]






# Classification CNN Stack 4
# C_CNN_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
C_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [256], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [512], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]}
]






# Classification CNN Stack 4 with MaxPooling
# C_CNN_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
C_CNN_S4_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},

    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]}
]






# Network Variations
CNN: ICNN = {
    "C_CNN_S1": C_CNN_S1,
    "C_CNN_S1_MP": C_CNN_S1_MP,
    "C_CNN_S2": C_CNN_S2,
    "C_CNN_S2_MP": C_CNN_S2_MP,
    "C_CNN_S3": C_CNN_S3,
    "C_CNN_S3_MP": C_CNN_S3_MP,
    "C_CNN_S4": C_CNN_S4,
    "C_CNN_S4_MP": C_CNN_S4_MP
}