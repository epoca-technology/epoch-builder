from typing import List, TypedDict
from modules.types import IKerasModelConfig



###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################


# Network Type
class ICLSTM(TypedDict):
    R_CLSTM_S1: List[IKerasModelConfig]
    R_CLSTM_S1_MP_DO: List[IKerasModelConfig]
    R_CLSTM_S2: List[IKerasModelConfig]
    R_CLSTM_S2_MP_DO: List[IKerasModelConfig]
    R_CLSTM_S3: List[IKerasModelConfig]
    R_CLSTM_S3_MP_DO: List[IKerasModelConfig]
    R_CLSTM_S4: List[IKerasModelConfig]
    R_CLSTM_S4_MP_DO: List[IKerasModelConfig]







# Regression CLSTM Stack 1
# R_CLSTM_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
R_CLSTM_S1: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512], "activations": [""]},
]







# Regression CLSTM Stack 1 with MaxPooling and Dropout
# R_CLSTM_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 dropout_rates:  Dropout_1
# 1 activations:    Conv1D_1
R_CLSTM_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32], "dropout_rates": [0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64], "dropout_rates": [0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128], "dropout_rates": [0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256], "dropout_rates": [0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "dropout_rates": [0], "activations": [""]},
]







# Regression CLSTM Stack 2
# R_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
R_CLSTM_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512], "activations": [""]}
]







# Regression CLSTM Stack 2 with MaxPooling and Dropout
# R_CLSTM_S2_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 1 activations:    Conv1D_1
R_CLSTM_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "dropout_rates": [0, 0], "activations": [""]}
]







# Regression CLSTM Stack 3
# R_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
R_CLSTM_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512], "activations": [""]}
]







# Regression CLSTM Stack 3 with MaxPooling and Dropout
# R_CLSTM_S3_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 1 activations:    Conv1D_1
R_CLSTM_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": [""]}
]







# Regression CLSTM Stack 4
# R_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
R_CLSTM_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": [""]},
]






# Regression CLSTM Stack 4 with MaxPooling and Dropout
# R_CLSTM_S4_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
# 1 activations:    Conv1D_1
R_CLSTM_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": [""]}
]









# Network Variations
CLSTM: ICLSTM = {
    "R_CLSTM_S1": R_CLSTM_S1,
    "R_CLSTM_S1_MP_DO": R_CLSTM_S1_MP_DO,
    "R_CLSTM_S2": R_CLSTM_S2,
    "R_CLSTM_S2_MP_DO": R_CLSTM_S2_MP_DO,
    "R_CLSTM_S3": R_CLSTM_S3,
    "R_CLSTM_S3_MP_DO": R_CLSTM_S3_MP_DO,
    "R_CLSTM_S4": R_CLSTM_S4,
    "R_CLSTM_S4_MP_DO": R_CLSTM_S4_MP_DO,
}