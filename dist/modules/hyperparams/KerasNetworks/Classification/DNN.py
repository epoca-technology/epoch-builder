from typing import List
from modules._types import IKerasModelConfig, IKerasHyperparamsNetworkVariations


#########################
## Deep Neural Network ##
#########################




# Classification DNN Stack 2
# KC_DNN_S2 
# 2 units:          Dense_1, Dense_2
# 2 activations:    Dense_1, Dense_2
KC_DNN_S2: List[IKerasModelConfig] = [
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
# KC_DNN_S3
# 3 units:          Dense_1, Dense_2, Dense_3
# 3 activations:    Dense_1, Dense_2, Dense_3
KC_DNN_S3: List[IKerasModelConfig] = [
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
# KC_DNN_S4
# 4 units:          Dense_1, Dense_2, Dense_3, Dense_4
# 4 activations:    Dense_1, Dense_2, Dense_3, Dense_4
KC_DNN_S4: List[IKerasModelConfig] = [
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


# Classification DNN Stack 5
# KC_DNN_S5
# 5 units:          Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
# 5 activations:    Dense_1, Dense_2, Dense_3, Dense_4, Dense_5
KC_DNN_S5: List[IKerasModelConfig] = [
    {"units": [32, 32, 32, 32, 32], "activations": ["", "", "", "", ""]},

    {"units": [64, 32, 32, 32, 32], "activations": ["", "", "", "", ""]},
    {"units": [64, 64, 64, 64, 64], "activations": ["", "", "", "", ""]},

    {"units": [128, 64, 64, 32, 32], "activations": ["", "", "", "", ""]},
    {"units": [128, 64, 64, 64, 64], "activations": ["", "", "", "", ""]},
    {"units": [128, 128, 128, 128, 128], "activations": ["", "", "", "", ""]},

    {"units": [256, 128, 64, 32, 32], "activations": ["", "", "", "", ""]},
    {"units": [256, 128, 128, 64, 64], "activations": ["", "", "", "", ""]},
    {"units": [256, 128, 128, 128, 128], "activations": ["", "", "", "", ""]},
    {"units": [256, 256, 256, 256, 256], "activations": ["", "", "", "", ""]},

    {"units": [512, 256, 128, 64, 32], "activations": ["", "", "", "", ""]},
    {"units": [512, 256, 256, 128, 128], "activations": ["", "", "", "", ""]},
    {"units": [512, 256, 256, 256, 256], "activations": ["", "", "", "", ""]},
    {"units": [512, 512, 512, 512, 512], "activations": ["", "", "", "", ""]}
]



# Network Variations
DNN: IKerasHyperparamsNetworkVariations = {
    "KC_DNN_S2": KC_DNN_S2,
    "KC_DNN_S3": KC_DNN_S3,
    "KC_DNN_S4": KC_DNN_S4,
    "KC_DNN_S5": KC_DNN_S5
}