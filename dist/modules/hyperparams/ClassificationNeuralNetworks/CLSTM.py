from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig



###################################################################
## Convolutional Long Short-Term Memory Recurrent Neural Network ##
###################################################################


# Network Type
class ICLSTM(TypedDict):
    C_CLSTM_S1: List[IKerasModelConfig]
    C_CLSTM_S1_DO: List[IKerasModelConfig]
    C_CLSTM_S1_MP: List[IKerasModelConfig]
    C_CLSTM_S1_MP_DO: List[IKerasModelConfig]
    C_CLSTM_S2: List[IKerasModelConfig]
    C_CLSTM_S2_DO: List[IKerasModelConfig]
    C_CLSTM_S2_MP: List[IKerasModelConfig]
    C_CLSTM_S2_MP_DO: List[IKerasModelConfig]
    C_CLSTM_S3: List[IKerasModelConfig]
    C_CLSTM_S3_DO: List[IKerasModelConfig]
    C_CLSTM_S3_MP: List[IKerasModelConfig]
    C_CLSTM_S3_MP_DO: List[IKerasModelConfig]
    C_CLSTM_S4: List[IKerasModelConfig]
    C_CLSTM_S4_DO: List[IKerasModelConfig]
    C_CLSTM_S4_MP: List[IKerasModelConfig]
    C_CLSTM_S4_MP_DO: List[IKerasModelConfig]







# Classification CLSTM Stack 1
# C_CLSTM_S1
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
C_CLSTM_S1: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512], "activations": [""]},
]




# Classification CLSTM Stack 1 with Dropout
# C_CLSTM_S1_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 units:          LSTM_1
# 1 dropout_rates:  Dropout_1
# 1 activations:    Conv1D_1
C_CLSTM_S1_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32], "dropout_rates": [0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64], "dropout_rates": [0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128], "dropout_rates": [0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256], "dropout_rates": [0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512], "dropout_rates": [0], "activations": [""]},
]




# Classification CLSTM Stack 1 with MaxPooling
# C_CLSTM_S1_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 activations:    Conv1D_1
C_CLSTM_S1_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "activations": [""]},
]







# Classification CLSTM Stack 1 with MaxPooling and Dropout
# C_CLSTM_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 dropout_rates:  Dropout_1
# 1 activations:    Conv1D_1
C_CLSTM_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32], "dropout_rates": [0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64], "dropout_rates": [0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128], "dropout_rates": [0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256], "dropout_rates": [0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512], "dropout_rates": [0], "activations": [""]},
]







# Classification CLSTM Stack 2
# C_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
C_CLSTM_S2: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512, 512], "activations": [""]}
]




# Classification CLSTM Stack 2 with Dropout
# C_CLSTM_S2_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 1 activations:    Conv1D_1
C_CLSTM_S2_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [64, 64], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256, 256], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512, 512], "dropout_rates": [0, 0], "activations": [""]}
]




# Classification CLSTM Stack 2 with MaxPooling
# C_CLSTM_S2_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 1 activations:    Conv1D_1
C_CLSTM_S2_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "activations": [""]}
]







# Classification CLSTM Stack 2 with MaxPooling and Dropout
# C_CLSTM_S2_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
# 1 activations:    Conv1D_1
C_CLSTM_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512], "dropout_rates": [0, 0], "activations": [""]}
]







# Classification CLSTM Stack 3
# C_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
C_CLSTM_S3: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256, 256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512, 512, 512], "activations": [""]}
]



# Classification CLSTM Stack 3 with Dropout
# C_CLSTM_S3_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 1 activations:    Conv1D_1
C_CLSTM_S3_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": [""]}
]




# Classification CLSTM Stack 3 with MaxPooling
# C_CLSTM_S3_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 1 activations:    Conv1D_1
C_CLSTM_S3_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "activations": [""]}
]





# Classification CLSTM Stack 3 with MaxPooling and Dropout
# C_CLSTM_S3_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
# 1 activations:    Conv1D_1
C_CLSTM_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": [""]}
]







# Classification CLSTM Stack 4
# C_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
C_CLSTM_S4: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 128, 64, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256, 256, 256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 256, 128, 64], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512, 512, 512, 512], "activations": [""]},
]




# Classification CLSTM Stack 4 with Dropout
# C_CLSTM_S4_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
# 1 activations:    Conv1D_1
C_CLSTM_S4_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "units": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "units": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
]





# Classification CLSTM Stack 4 with MaxPooling
# C_CLSTM_S4_MP
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 1 activations:    Conv1D_1
C_CLSTM_S4_MP: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 128, 64, 32], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "activations": [""]}
]







# Classification CLSTM Stack 4 with MaxPooling and Dropout
# C_CLSTM_S4_MP_DO
# 1 filters:        Conv1D_1
# 1 kernel_sizes:   Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
# 1 activations:    Conv1D_1
C_CLSTM_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [32], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [64], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [128], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [256], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [512], "kernel_sizes": [3], "pool_sizes": [2], "units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": [""]}
]









# Network Variations
CLSTM: ICLSTM = {
    "C_CLSTM_S1": C_CLSTM_S1,
    "C_CLSTM_S1_DO": C_CLSTM_S1_DO,
    "C_CLSTM_S1_MP": C_CLSTM_S1_MP,
    "C_CLSTM_S1_MP_DO": C_CLSTM_S1_MP_DO,
    "C_CLSTM_S2": C_CLSTM_S2,
    "C_CLSTM_S2_DO": C_CLSTM_S2_DO,
    "C_CLSTM_S2_MP": C_CLSTM_S2_MP,
    "C_CLSTM_S2_MP_DO": C_CLSTM_S2_MP_DO,
    "C_CLSTM_S3": C_CLSTM_S3,
    "C_CLSTM_S3_DO": C_CLSTM_S3_DO,
    "C_CLSTM_S3_MP": C_CLSTM_S3_MP,
    "C_CLSTM_S3_MP_DO": C_CLSTM_S3_MP_DO,
    "C_CLSTM_S4": C_CLSTM_S4,
    "C_CLSTM_S4_DO": C_CLSTM_S4_DO,
    "C_CLSTM_S4_MP": C_CLSTM_S4_MP,
    "C_CLSTM_S4_MP_DO": C_CLSTM_S4_MP_DO,
}