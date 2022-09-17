from typing import List
from modules._types import IKerasModelConfig, IRegressionTrainingConfigNetworks



###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################



# Regression CLSTM Stack 2
# KR_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
KR_CLSTM_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128], "activations": [""]}
]







# Regression CLSTM Stack 2 with MaxPooling
# KR_CLSTM_MP_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
KR_CLSTM_MP_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128], "activations": [""]}
]







# Regression CLSTM Stack 3
# KR_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
KR_CLSTM_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 128, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 128, 32], "activations": [""]}
]








# Regression CLSTM Stack 3 with MaxPooling
# KR_CLSTM_MP_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
KR_CLSTM_MP_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 128, 32], "activations": [""]}
]












# Regression CLSTM Stack 4
# KR_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
KR_CLSTM_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": [""]}
]








# Regression CLSTM Stack 4 with MaxPooling
# KR_CLSTM_MP_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
KR_CLSTM_MP_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": [""]}
]









# Regression CLSTM Stack 5
# KR_CLSTM_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
KR_CLSTM_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [512, 256, 128, 64, 32], "activations": [""]}
]







# Regression CLSTM Stack 5 with MaxPooling
# KR_CLSTM_MP_S5
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 5 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
# 1 activations:    Conv1D_1
KR_CLSTM_MP_S5: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256, 256], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64, 32], "activations": [""]}
]







# Network Variations
CLSTM: IRegressionTrainingConfigNetworks = {
    "KR_CLSTM_S2": KR_CLSTM_S2,
    "KR_CLSTM_MP_S2": KR_CLSTM_MP_S2,
    "KR_CLSTM_S3": KR_CLSTM_S3,
    "KR_CLSTM_MP_S3": KR_CLSTM_MP_S3,
    "KR_CLSTM_S4": KR_CLSTM_S4,
    "KR_CLSTM_MP_S4": KR_CLSTM_MP_S4,
    "KR_CLSTM_S5": KR_CLSTM_S5,
    "KR_CLSTM_MP_S5": KR_CLSTM_MP_S5
}