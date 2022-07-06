from typing import List, TypedDict
from modules.types import IKerasModelConfig


#########################
## Deep Neural Network ##
#########################


# Network Type
class IDNN(TypedDict):
    C_DNN_S1: List[IKerasModelConfig]
    C_DNN_S2: List[IKerasModelConfig]
    C_DNN_S3: List[IKerasModelConfig]
    C_DNN_S4: List[IKerasModelConfig]





# Classification DNN Stack 1
# C_DNN_S1
# 1 units:       Dense_1
# 1 activations: Dense_1
C_DNN_S1: List[IKerasModelConfig] = [
    {"units": [32], "activations": [""]},
    {"units": [64], "activations": [""]},
    {"units": [128], "activations": [""]},
    {"units": [256], "activations": [""]},
    {"units": [512], "activations": [""]}
]




# Classification DNN Stack 2
# C_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
C_DNN_S2: List[IKerasModelConfig] = [
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









# Classification DNN Stack 3
# C_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
C_DNN_S3: List[IKerasModelConfig] = [
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









# Classification DNN Stack 4
# C_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
C_DNN_S4: List[IKerasModelConfig] = [
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















# Network Variations
DNN: IDNN = {
    "C_DNN_S1": C_DNN_S1,
    "C_DNN_S2": C_DNN_S2,
    "C_DNN_S3": C_DNN_S3,
    "C_DNN_S4": C_DNN_S4
}