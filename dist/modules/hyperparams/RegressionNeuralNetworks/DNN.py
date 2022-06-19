from typing import List, TypedDict
from modules.keras_models import IKerasModelConfig


#########################
## Deep Neural Network ##
#########################


# Network Type
class IDNN(TypedDict):
    R_DNN_S1: List[IKerasModelConfig]
    R_DNN_S2: List[IKerasModelConfig]
    R_DNN_S2_DO: List[IKerasModelConfig]
    R_DNN_S3: List[IKerasModelConfig]
    R_DNN_S3_DO: List[IKerasModelConfig]
    R_DNN_S4: List[IKerasModelConfig]
    R_DNN_S4_DO: List[IKerasModelConfig]





# Regression DNN Stack 1
# R_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
R_DNN_S1: List[IKerasModelConfig] = [
    {"units": [32], "activations": [""]},
    {"units": [64], "activations": [""]},
    {"units": [128], "activations": [""]},
    {"units": [256], "activations": [""]},
    {"units": [512], "activations": [""]}
]




# Regression DNN Stack 2
# R_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
R_DNN_S2: List[IKerasModelConfig] = [
    {"units": [32, 32], "activations": ["", ""]},

    {"units": [64, 32], "activations": ["", ""]},
    {"units": [64, 64], "activations": ["", ""]},

    {"units": [128, 64], "activations": ["", ""]},
    {"units": [128, 128], "activations": ["", ""]},

    {"units": [256, 64], "activations": ["", ""]},
    {"units": [256, 128], "activations": ["", ""]},
    {"units": [256, 256], "activations": ["", ""]},

    {"units": [512, 128], "activations": ["", ""]},
    {"units": [512, 256], "activations": ["", ""]},
    {"units": [512, 512], "activations": ["", ""]}
]





# Regression DNN Stack 2 with Dropout
# R_DNN_S2_DO
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
# 2 dropout_rates:  Dropout_1, Dropout_2
R_DNN_S2_DO: List[IKerasModelConfig] = [
    {"units": [32, 32], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"units": [64, 32], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"units": [64, 64], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"units": [128, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"units": [128, 128], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"units": [256, 64], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"units": [256, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"units": [256, 256], "dropout_rates": [0, 0], "activations": ["", ""]},

    {"units": [512, 128], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"units": [512, 256], "dropout_rates": [0, 0], "activations": ["", ""]},
    {"units": [512, 512], "dropout_rates": [0, 0], "activations": ["", ""]}
]





# Regression DNN Stack 3
# R_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
R_DNN_S3: List[IKerasModelConfig] = [
    {"units": [32, 32, 32], "activations": ["", "", ""]},

    {"units": [64, 32, 32], "activations": ["", "", ""]},
    {"units": [64, 64, 64], "activations": ["", "", ""]},

    {"units": [128, 64, 32], "activations": ["", "", ""]},
    {"units": [128, 64, 64], "activations": ["", "", ""]},
    {"units": [128, 128, 128], "activations": ["", "", ""]},

    {"units": [256, 64, 32], "activations": ["", "", ""]},
    {"units": [256, 128, 64], "activations": ["", "", ""]},
    {"units": [256, 128, 128], "activations": ["", "", ""]},
    {"units": [256, 256, 256], "activations": ["", "", ""]},

    {"units": [512, 128, 64], "activations": ["", "", ""]},
    {"units": [512, 256, 128], "activations": ["", "", ""]},
    {"units": [512, 256, 256], "activations": ["", "", ""]},
    {"units": [512, 512, 512], "activations": ["", "", ""]}
]





# Regression DNN Stack 3 with Dropout
# R_DNN_S3_DO
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
# 3 dropout_rates:  Dropout_1, Dropout_2, Dropout_3
R_DNN_S3_DO: List[IKerasModelConfig] = [
    {"units": [32, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [64, 32, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [64, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [128, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [128, 64, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [128, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [256, 64, 32], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [256, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [256, 128, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [256, 256, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},

    {"units": [512, 128, 64], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [512, 256, 128], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [512, 256, 256], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]},
    {"units": [512, 512, 512], "dropout_rates": [0, 0, 0], "activations": ["", "", ""]}
]




# Regression DNN Stack 4
# R_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
R_DNN_S4: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32], "activations": ["", "", "", ""]},

    {"units": [64, 32, 32, 32], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 64], "activations": ["", "", "", ""]},

    {"units": [128, 64, 32, 32], "activations": ["", "", "", ""]},
    {"units": [128, 64, 64, 64], "activations": ["", "", "", ""]},
    {"units": [128, 128, 128, 128], "activations": ["", "", "", ""]},

    {"units": [256, 128, 64, 32], "activations": ["", "", "", ""]},
    {"units": [256, 128, 128, 64], "activations": ["", "", "", ""]},
    {"units": [256, 128, 128, 128], "activations": ["", "", "", ""]},
    {"units": [256, 256, 256, 256], "activations": ["", "", "", ""]},

    {"units": [512, 256, 128, 64], "activations": ["", "", "", ""]},
    {"units": [512, 256, 256, 128], "activations": ["", "", "", ""]},
    {"units": [512, 256, 256, 256], "activations": ["", "", "", ""]},
    {"units": [512, 512, 512, 512], "activations": ["", "", "", ""]}
]






# Regression DNN Stack 4 with Dropout
# R_DNN_S4_DO
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
# 4 dropout_rates:  Dropout_1, Dropout_2, Dropout_3, Dropout_4
R_DNN_S4_DO: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [64, 32, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [64, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [128, 64, 32, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [128, 64, 64, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [128, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [256, 128, 64, 32], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [256, 128, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [256, 128, 128, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [256, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},

    {"units": [512, 256, 128, 64], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [512, 256, 256, 128], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [512, 256, 256, 256], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]},
    {"units": [512, 512, 512, 512], "dropout_rates": [0, 0, 0, 0], "activations": ["", "", "", ""]}
]








# Network Variations
DNN: IDNN = {
    "R_DNN_S1": R_DNN_S1,
    "R_DNN_S2": R_DNN_S2,
    "R_DNN_S2_DO": R_DNN_S2_DO,
    "R_DNN_S3": R_DNN_S3,
    "R_DNN_S4": R_DNN_S4,
    "R_DNN_S4_DO": R_DNN_S4_DO
}