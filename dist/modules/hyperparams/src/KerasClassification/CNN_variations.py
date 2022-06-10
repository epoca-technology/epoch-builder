from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


## Convolutional Neural Network ##


# Network Type
class ICNN(TypedDict):
    C_CNN_S1: List[IKerasModelConfig]
    C_CNN_S1_MP_DO: List[IKerasModelConfig]
    C_CNN_S2: List[IKerasModelConfig]
    C_CNN_S2_MP_DO: List[IKerasModelConfig]
    C_CNN_S3: List[IKerasModelConfig]
    C_CNN_S3_MP_DO: List[IKerasModelConfig]
    C_CNN_S4: List[IKerasModelConfig]
    C_CNN_S4_MP_DO: List[IKerasModelConfig]





# Classification CNN Stack 1
# C_CNN_S1
# 1 filters:     Conv1D_1
# 1 activations: Conv1D_1
C_CNN_S1: List[IKerasModelConfig] = [
    {"filters": [16], "activations": [""]},
    {"filters": [32], "activations": [""]},
    {"filters": [64], "activations": [""]},
    {"filters": [128], "activations": [""]}
]






# Classification CNN Stack 1 with MaxPooling and Dropout
# C_CNN_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 dropout_rates:  Dropout_1
C_CNN_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [32], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [64], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [128], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]}
]





# Classification CNN Stack 2
# C_CNN_S2
# 2 filters:     Conv1D_1, Conv1D_2
# 2 activations: Conv1D_1, Conv1D_2
C_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [16, 16], "activations": ["", ""]},

    {"filters": [32, 16], "activations": ["", ""]},
    {"filters": [32, 32], "activations": ["", ""]},

    {"filters": [64, 32], "activations": ["", ""]},
    {"filters": [64, 64], "activations": ["", ""]},

    {"filters": [128, 64], "activations": ["", ""]},
    {"filters": [128, 128], "activations": ["", ""]}
]






# Classification CNN Stack 2 with MaxPooling and Dropout
# C_CNN_S2_MP_DO
# 2 filters:        Conv1D_1, Conv1D_2
# 2 activations:    Conv1D_1, Conv1D_2
# 2 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_CNN_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [32, 16], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [32, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [64, 32], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [64, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},

    {"filters": [128, 64], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]},
    {"filters": [128, 128], "dropout_rates": [0, 0], "pool_sizes": [0, 0], "activations": ["", ""]}
]




# Classification CNN Stack 3
# C_CNN_S3
# 3 filters:     Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations: Conv1D_1, Conv1D_2, Conv1D_3
C_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [16, 16, 16], "activations": ["", "", ""]},
    
    {"filters": [32, 16, 16], "activations": ["", "", ""]},
    {"filters": [32, 32, 32], "activations": ["", "", ""]},
    
    {"filters": [64, 32, 32], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "activations": ["", "", ""]},

    {"filters": [64, 32, 16], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "activations": ["", "", ""]},

    {"filters": [128, 32, 16], "activations": ["", "", ""]}
]






# Classification CNN Stack 3 with MaxPooling and Dropout
# C_CNN_S3_MP_DO
# 3 filters:        Conv1D_1, Conv1D_2, Conv1D_3
# 3 activations:    Conv1D_1, Conv1D_2, Conv1D_3
# 3 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_CNN_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16, 16, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [32, 16, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [32, 32, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    
    {"filters": [64, 32, 32], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [64, 64, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [64, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 64, 64], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},
    {"filters": [128, 128, 128], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]},

    {"filters": [128, 32, 16], "dropout_rates": [0, 0, 0], "pool_sizes": [0, 0, 0], "activations": ["", "", ""]}
]






# Classification CNN Stack 4
# C_CNN_S4
# 4 filters:     Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations: Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
C_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [16, 16, 16, 16], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 16, 16], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 32], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 16, 16], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 16], "activations": ["", "", "", ""]}
]







# Classification CNN Stack 4 with MaxPooling and Dropout
# C_CNN_S4_MP_DO
# 4 filters:        Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 activations:    Conv1D_1, Conv1D_2, Conv1D_3, Conv1D_4
# 4 pool_sizes:     MaxPooling1D_1, MaxPooling1D_2, MaxPooling1D_3, MaxPooling1D_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_CNN_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [32, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [64, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"filters": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"filters": [128, 64, 32, 16], "dropout_rates": [0, 0, 0, 0], "pool_sizes": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]



# Network Variations
CNN: ICNN = {
    "C_CNN_S1": C_CNN_S1,
    "C_CNN_S1_MP_DO": C_CNN_S1_MP_DO,
    "C_CNN_S2": C_CNN_S2,
    "C_CNN_S2_MP_DO": C_CNN_S2_MP_DO,
    "C_CNN_S3": C_CNN_S3,
    "C_CNN_S3_MP_DO": C_CNN_S3_MP_DO,
    "C_CNN_S4": C_CNN_S4,
    "C_CNN_S4_MP_DO": C_CNN_S4_MP_DO
}