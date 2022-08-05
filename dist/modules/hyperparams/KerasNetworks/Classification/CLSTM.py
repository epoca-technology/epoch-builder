from typing import List
from modules.types import IKerasModelConfig, IKerasHyperparamsNetworkVariations



###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################




# Classification CLSTM Stack 2
# KC_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
KC_CLSTM_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128], "activations": [""]}
]





# Classification CLSTM Stack 2 with MaxPooling
# KC_CLSTM_S2_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
KC_CLSTM_S2_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": [""]}
]






# Classification CLSTM Stack 3
# KC_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
KC_CLSTM_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 128, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128, 32], "activations": [""]}
]






# Classification CLSTM Stack 3 with MaxPooling
# KC_CLSTM_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
KC_CLSTM_S3_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": [""]}
]







# Classification CLSTM Stack 4
# KC_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
KC_CLSTM_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": [""]}
]







# Classification CLSTM Stack 4 with MaxPooling
# KC_CLSTM_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
KC_CLSTM_S4_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": [""]}
]




# Classification CLSTM Stack 5
# KC_CLSTM_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
KC_CLSTM_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 512, 512, 512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": [""]}
]




# Classification CLSTM Stack 5 with MaxPooling
# KC_CLSTM_S5_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
KC_CLSTM_S5_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512, 512], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": [""]}
]





# Network Variations
CLSTM: IKerasHyperparamsNetworkVariations = {
    "KC_CLSTM_S2": KC_CLSTM_S2,
    "KC_CLSTM_S2_MP": KC_CLSTM_S2_MP,
    "KC_CLSTM_S3": KC_CLSTM_S3,
    "KC_CLSTM_S3_MP": KC_CLSTM_S3_MP,
    "KC_CLSTM_S4": KC_CLSTM_S4,
    "KC_CLSTM_S4_MP": KC_CLSTM_S4_MP,
    "KC_CLSTM_S5": KC_CLSTM_S5,
    "KC_CLSTM_S5_MP": KC_CLSTM_S5_MP
}