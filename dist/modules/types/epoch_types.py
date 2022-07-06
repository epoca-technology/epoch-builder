from typing import TypedDict, Union




# Arima Config
# The configuration to be used on the Arima Instance to generate predictions.
class IEpochConfig(TypedDict):
    # Identifier, must be preffixed with "_EPOCHNAME"
    id: str

    # The range of the Epoch
    start: int
    end: int

    # The range that will be used to train the KerasModels
    training_start: int
    training_end: int


