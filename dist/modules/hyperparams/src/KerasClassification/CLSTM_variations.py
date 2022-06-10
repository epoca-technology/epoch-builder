from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


## Convolutional Long Short-Term Memory Recurrent Neural Network ##


# Network Type
class ICLSTM(TypedDict):
    C_CLSTM_S1: List[IKerasModelConfig]
    C_CLSTM_S1_MP_DO: List[IKerasModelConfig]
    C_CLSTM_S2: List[IKerasModelConfig]
    C_CLSTM_S2_MP_DO: List[IKerasModelConfig]
    C_CLSTM_S3: List[IKerasModelConfig]
    C_CLSTM_S3_MP_DO: List[IKerasModelConfig]
    C_CLSTM_S4: List[IKerasModelConfig]
    C_CLSTM_S4_MP_DO: List[IKerasModelConfig]





# Classification CLSTM Stack 1
# C_CLSTM_S1
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 units:          LSTM_1
C_CLSTM_S1: List[IKerasModelConfig] = [
    {"filters": [32], "units": [32], "activations": [""]},
    {"filters": [64], "units": [64], "activations": [""]},
    {"filters": [128], "units": [128], "activations": [""]},
]






# Classification CLSTM Stack 1 with MaxPooling and Dropout
# C_CLSTM_S1_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 1 units:          LSTM_1
# 1 dropout_rates:  Dropout_1
C_CLSTM_S1_MP_DO: List[IKerasModelConfig] = [
    {"filters": [32], "units": [32], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [64], "units": [64], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
    {"filters": [128], "units": [128], "dropout_rates": [0], "pool_sizes": [0], "activations": [""]},
]





# Classification CLSTM Stack 2
# C_CLSTM_S2
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 2 units:          LSTM_1, LSTM_2
C_CLSTM_S2: List[IKerasModelConfig] = [
    {"filters": [16], "units": [16, 16], "activations": [""]},

    {"filters": [32], "units": [16, 16], "activations": [""]},
    {"filters": [32], "units": [32, 32], "activations": [""]},

    {"filters": [64], "units": [32, 32], "activations": [""]},
    {"filters": [64], "units": [64, 64], "activations": [""]},
    
    {"filters": [128], "units": [64, 64], "activations": [""]},
    {"filters": [128], "units": [128, 128], "activations": [""]}
]








# Classification CLSTM Stack 2 with MaxPooling and Dropout
# C_CLSTM_S2_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 2 units:          LSTM_1, LSTM_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_CLSTM_S2_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16], "units": [16, 16], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [32], "units": [16, 16], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [32], "units": [32, 32], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [64], "units": [32, 32], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [64], "units": [64, 64], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]},

    {"filters": [128], "units": [64, 64], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]},
    {"filters": [128], "units": [128, 128], "pool_sizes": [0], "dropout_rates": [0, 0], "activations": [""]}
]







# Classification CLSTM Stack 3
# C_CLSTM_S3
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
C_CLSTM_S3: List[IKerasModelConfig] = [
    {"filters": [16], "units": [16, 16, 16], "activations": [""]},

    {"filters": [32], "units": [16, 16, 16], "activations": [""]},
    {"filters": [32], "units": [32, 32, 32], "activations": [""]},

    {"filters": [64], "units": [32, 32, 32], "activations": [""]},
    {"filters": [64], "units": [64, 64, 64], "activations": [""]},

    {"filters": [128], "units": [64, 64, 64], "activations": [""]},
    {"filters": [128], "units": [128, 128, 128], "activations": [""]},
]







# Classification CLSTM Stack 3 with MaxPooling and Dropout
# C_CLSTM_S3_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 3 units:          LSTM_1, LSTM_2, LSTM_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_CLSTM_S3_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16], "units": [16, 16, 16], "activations": [""]},

    {"filters": [32], "pool_sizes": [0], "units": [16, 16, 16], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [32], "pool_sizes": [0], "units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [64], "pool_sizes": [0], "units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [64], "pool_sizes": [0], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},

    {"filters": [128], "pool_sizes": [0], "units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": [""]},
    {"filters": [128], "pool_sizes": [0], "units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": [""]},
]







# Classification CLSTM Stack 4
# C_CLSTM_S4
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
C_CLSTM_S4: List[IKerasModelConfig] = [
    {"filters": [16], "units": [16, 16, 16, 16], "activations": [""]},

    {"filters": [32], "units": [16, 16, 16, 16], "activations": [""]},
    {"filters": [32], "units": [32, 32, 32, 32], "activations": [""]},

    {"filters": [64], "units": [32, 32, 32, 32], "activations": [""]},
    {"filters": [64], "units": [64, 64, 64, 64], "activations": [""]},

    {"filters": [128], "units": [64, 64, 64, 64], "activations": [""]},
    {"filters": [128], "units": [128, 128, 128, 128], "activations": [""]}
]







# Classification CLSTM Stack 4 with MaxPooling and Dropout
# C_CLSTM_S4_MP_DO
# 1 filters:        Conv1D_1
# 1 activations:    Conv1D_1
# 1 pool_sizes:     MaxPooling1D_1
# 4 units:          LSTM_1, LSTM_2, LSTM_3, LSTM_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_CLSTM_S4_MP_DO: List[IKerasModelConfig] = [
    {"filters": [16], "units": [16, 16, 16, 16], "activations": [""]},

    {"filters": [32], "pool_sizes": [0], "units": [16, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [32], "pool_sizes": [0], "units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [64], "pool_sizes": [0], "units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [64], "pool_sizes": [0], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},

    {"filters": [128], "pool_sizes": [0], "units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": [""]},
    {"filters": [128], "pool_sizes": [0], "units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": [""]}
]









# Network Variations
CLSTM: ICLSTM = {
    "C_CLSTM_S1": C_CLSTM_S1,
    "C_CLSTM_S1_MP_DO": C_CLSTM_S1_MP_DO,
    "C_CLSTM_S2": C_CLSTM_S2,
    "C_CLSTM_S2_MP_DO": C_CLSTM_S2_MP_DO,
    "C_CLSTM_S3": C_CLSTM_S3,
    "C_CLSTM_S3_MP_DO": C_CLSTM_S3_MP_DO,
    "C_CLSTM_S4": C_CLSTM_S4,
    "C_CLSTM_S4_MP_DO": C_CLSTM_S4_MP_DO,
}