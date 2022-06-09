from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


## Deep Neural Network ##


# Network Type
class IDNN(TypedDict):
    C_DNN_S1: List[IKerasModelConfig]
    C_DNN_S2: List[IKerasModelConfig]
    C_DNN_S2_DO: List[IKerasModelConfig]
    C_DNN_S3: List[IKerasModelConfig]
    C_DNN_S3_DO: List[IKerasModelConfig]
    C_DNN_S4: List[IKerasModelConfig]
    C_DNN_S4_DO: List[IKerasModelConfig]





# Classification DNN Stack 1
# C_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
C_DNN_S1: List[IKerasModelConfig] = [
    {"units": [16], "activations": [""]},
    {"units": [32], "activations": [""]},
    {"units": [64], "activations": [""]},
    {"units": [128], "activations": [""]}
]




# Classification DNN Stack 2
# C_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
C_DNN_S2: List[IKerasModelConfig] = [
    {"units": [16, 16], "activations": ["", ""]},

    {"units": [32, 16], "activations": ["", ""]},
    {"units": [16, 32], "activations": ["", ""]},
    {"units": [32, 32], "activations": ["", ""]},

    {"units": [64, 32], "activations": ["", ""]},
    {"units": [32, 64], "activations": ["", ""]},
    {"units": [64, 64], "activations": ["", ""]},
    
    {"units": [64, 16], "activations": ["", ""]},

    {"units": [64, 128], "activations": ["", ""]},
    {"units": [128, 64], "activations": ["", ""]},
    {"units": [128, 128], "activations": ["", ""]},

    {"units": [128, 16], "activations": ["", ""]}
]





# Classification DNN Stack 2 with Dropout
# C_DNN_S2_DO
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
C_DNN_S2_DO: List[IKerasModelConfig] = [
        {"units": [16, 16], "dropout_rates": [0, 0], "activations": ["", ""]},

        {"units": [32, 16], "dropout_rates": [0, 0], "activations": ["", ""]},
        {"units": [16, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
        {"units": [32, 32], "dropout_rates": [0, 0], "activations": ["", ""]},

        {"units": [64, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
        {"units": [32, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
        {"units": [64, 64], "dropout_rates": [0, 0], "activations": ["", ""]},

        {"units": [64, 16], "dropout_rates": [0, 0], "activations": ["", ""]},

        {"units": [64, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
        {"units": [128, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
        {"units": [128, 128], "dropout_rates": [0, 0], "activations": ["", ""]},

        {"units": [128, 16], "dropout_rates": [0, 0], "activations": ["", ""]}
    ]



# Classification DNN Stack 3
# C_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
C_DNN_S3: List[IKerasModelConfig] = [
    {"units": [16, 16, 16], "activations": ["", "", ""]},

    {"units": [32, 16, 16], "activations": ["", "", ""]},
    {"units": [16, 32, 16], "activations": ["", "", ""]},
    {"units": [16, 16, 32], "activations": ["", "", ""]},
    {"units": [32, 32, 16], "activations": ["", "", ""]},
    {"units": [16, 32, 32], "activations": ["", "", ""]},
    {"units": [32, 32, 32], "activations": ["", "", ""]},

    {"units": [64, 32, 32], "activations": ["", "", ""]},
    {"units": [32, 64, 32], "activations": ["", "", ""]},
    {"units": [32, 32, 64], "activations": ["", "", ""]},
    {"units": [64, 64, 32], "activations": ["", "", ""]},
    {"units": [32, 64, 64], "activations": ["", "", ""]},
    {"units": [64, 64, 64], "activations": ["", "", ""]},

    {"units": [64, 32, 16], "activations": ["", "", ""]},

    {"units": [128, 64, 64], "activations": ["", "", ""]},
    {"units": [64, 128, 64], "activations": ["", "", ""]},
    {"units": [64, 64, 128], "activations": ["", "", ""]},
    {"units": [128, 128, 64], "activations": ["", "", ""]},
    {"units": [64, 128, 128], "activations": ["", "", ""]},
    {"units": [128, 128, 128], "activations": ["", "", ""]},

    {"units": [128, 32, 16], "activations": ["", "", ""]}
]



# Classification DNN Stack 3 with Dropout
# C_DNN_S3_DO
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
C_DNN_S3_DO: List[IKerasModelConfig] = [
    {"units": [16, 16, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [32, 16, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [16, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [16, 16, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [32, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [16, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [64, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [32, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [32, 32, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [64, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [32, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [64, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [128, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [64, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [64, 64, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [128, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [64, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [128, 32, 16], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]}
]




# Classification DNN Stack 4
# C_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
C_DNN_S4: List[IKerasModelConfig] = [
    {"units": [16, 16, 16, 16], "activations": ["", "", "", ""]},

    {"units": [32, 16, 16, 16], "activations": ["", "", "", ""]},
    {"units": [16, 32, 16, 16], "activations": ["", "", "", ""]},
    {"units": [16, 16, 32, 16], "activations": ["", "", "", ""]},
    {"units": [16, 16, 16, 32], "activations": ["", "", "", ""]},
    {"units": [32, 32, 16, 16], "activations": ["", "", "", ""]},
    {"units": [16, 16, 32, 32], "activations": ["", "", "", ""]},
    {"units": [32, 32, 32, 16], "activations": ["", "", "", ""]},
    {"units": [16, 32, 32, 32], "activations": ["", "", "", ""]},
    {"units": [32, 32, 32, 32], "activations": ["", "", "", ""]},

    {"units": [64, 32, 32, 32], "activations": ["", "", "", ""]},
    {"units": [32, 64, 32, 32], "activations": ["", "", "", ""]},
    {"units": [32, 32, 64, 32], "activations": ["", "", "", ""]},
    {"units": [32, 32, 32, 64], "activations": ["", "", "", ""]},
    {"units": [64, 64, 32, 32], "activations": ["", "", "", ""]},
    {"units": [32, 32, 64, 64], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 32], "activations": ["", "", "", ""]},
    {"units": [32, 64, 64, 64], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 64], "activations": ["", "", "", ""]},

    {"units": [64, 32, 16, 16], "activations": ["", "", "", ""]},

    {"units": [128, 64, 64, 64], "activations": ["", "", "", ""]},
    {"units": [64, 128, 64, 64], "activations": ["", "", "", ""]},
    {"units": [64, 64, 128, 64], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 128], "activations": ["", "", "", ""]},
    {"units": [128, 128, 64, 64], "activations": ["", "", "", ""]},
    {"units": [64, 64, 128, 128], "activations": ["", "", "", ""]},
    {"units": [128, 128, 128, 64], "activations": ["", "", "", ""]},
    {"units": [64, 128, 128, 128], "activations": ["", "", "", ""]},
    {"units": [128, 128, 128, 128], "activations": ["", "", "", ""]},

    {"units": [128, 64, 32, 16], "activations": ["", "", "", ""]}
]






# Classification DNN Stack 4 with Dropout
# C_DNN_S4_DO
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
C_DNN_S4_DO: List[IKerasModelConfig] = [
    {"units": [16, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [32, 16, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [16, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [16, 16, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [16, 16, 16, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [16, 16, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 32, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [16, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 32, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 32, 32, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 32, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [32, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [64, 32, 16, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [128, 128, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [128, 128, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [128, 64, 32, 16], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]








# Network Variations
DNN: IDNN = {
    "C_DNN_S1": C_DNN_S1,
    "C_DNN_S2": C_DNN_S2,
    "C_DNN_S2_DO": C_DNN_S2_DO,
    "C_DNN_S3": C_DNN_S3,
    "C_DNN_S4": C_DNN_S4,
    "C_DNN_S4_DO": C_DNN_S4_DO
}