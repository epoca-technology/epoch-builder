from typing import List
from modules._types import IKerasModelConfig, IKerasHyperparamsNetworkVariations


##################################
## Convolutional Neural Network ##
##################################





# Classification CNN Stack 2
# KC_CNN_S2
# 1 filters:        Conv1D_1, Conv1D_2
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
KC_CNN_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128], "activations": ["", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128], "activations": ["", "", ""]}
]





# Classification CNN Stack 2 with Dropout
# KC_CNN_DO_S2
# 1 filters:        Conv1D_1, Conv1D_2
# 1 kernel_sizes:   Conv1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
KC_CNN_DO_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]}
]





# Classification CNN Stack 2 with MaxPooling
# KC_CNN_MP_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
KC_CNN_MP_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": ["", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": ["", "", ""]}
]





# Classification CNN Stack 2 with MaxPooling and Dropout
# KC_CNN_MP_DO_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          Dense_1, Dense_2
# 3 activations:    Conv1D_1, Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
KC_CNN_MP_DO_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": ["", "", ""], "dropout_rates": [0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": ["", "", ""], "dropout_rates": [0, 0]}
]






# Classification CNN Stack 3
# KC_CNN_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
KC_CNN_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128, 32], "activations": ["", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128, 32], "activations": ["", "", "", ""]}
]




# Classification CNN Stack 3 with Dropout
# KC_CNN_DO_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
KC_CNN_DO_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]}
]






# Classification CNN Stack 3 with MaxPooling
# KC_CNN_MP_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
KC_CNN_MP_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": ["", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": ["", "", "", ""]}
]






# Classification CNN Stack 3 with MaxPooling and Dropout
# KC_CNN_MP_DO_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          Dense_1, Dense_2, Dense_3
# 4 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
KC_CNN_MP_DO_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": ["", "", "", ""], "dropout_rates": [0, 0, 0]}
]






# Classification CNN Stack 4
# KC_CNN_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
KC_CNN_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]}
]






# Classification CNN Stack 4 with Dropout
# KC_CNN_DO_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
KC_CNN_DO_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]}
]





# Classification CNN Stack 4 with MaxPooling
# KC_CNN_MP_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
KC_CNN_MP_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""]}
]





# Classification CNN Stack 4 with MaxPooling and Dropout
# KC_CNN_MP_DO_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 5 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
KC_CNN_MP_DO_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": ["", "", "", "", ""], "dropout_rates": [0, 0, 0, 0]}
]





# Classification CNN Stack 5
# KC_CNN_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
KC_CNN_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]}
]







# Classification CNN Stack 5 with Dropout
# KC_CNN_DO_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
KC_CNN_DO_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]}
]




# Classification CNN Stack 5 with MaxPooling
# KC_CNN_MP_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
KC_CNN_MP_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""]}
]




# Classification CNN Stack 5 with MaxPooling and Dropout
# KC_CNN_MP_DO_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 6 activations:    Conv1D_1, Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4, Dropout_5
KC_CNN_MP_DO_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", "", ""], "dropout_rates": [0, 0, 0, 0, 0]}
]





# Network Variations
CNN: IKerasHyperparamsNetworkVariations = {
    "KC_CNN_S2": KC_CNN_S2,
    "KC_CNN_DO_S2": KC_CNN_DO_S2,
    "KC_CNN_MP_S2": KC_CNN_MP_S2,
    "KC_CNN_MP_DO_S2": KC_CNN_MP_DO_S2,
    "KC_CNN_S3": KC_CNN_S3,
    "KC_CNN_DO_S3": KC_CNN_DO_S3,
    "KC_CNN_MP_S3": KC_CNN_MP_S3,
    "KC_CNN_MP_DO_S3": KC_CNN_MP_DO_S3,
    "KC_CNN_S4": KC_CNN_S4,
    "KC_CNN_DO_S4": KC_CNN_DO_S4,
    "KC_CNN_MP_S4": KC_CNN_MP_S4,
    "KC_CNN_MP_DO_S4": KC_CNN_MP_DO_S4,
    "KC_CNN_S5": KC_CNN_S5,
    "KC_CNN_DO_S5": KC_CNN_DO_S5,
    "KC_CNN_MP_S5": KC_CNN_MP_S5,
    "KC_CNN_MP_DO_S5": KC_CNN_MP_DO_S5
}