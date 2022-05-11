from typing import TypedDict




## Percent Change Interpreter ##


# Percent Change Interpreter Configuration
# The configuration used in order to interpret the model's predictions based on the 
# percentual change from the first to the last prediction.
class IPercentChangeInterpreterConfig(TypedDict):
    long: float
    short: float






## Probability Interpreter ##


# Probability Interpreter Configuration
# The configuration used in order to interpret the model's predictions based on the 
# up and down probability.
class IProbabilityInterpreterConfig(TypedDict):
    min_probability: float