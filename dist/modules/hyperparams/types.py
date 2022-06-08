from typing import TypedDict




# Keras Loss
# ...
class IKerasLoss(TypedDict):
    func_name: str
    metric: str