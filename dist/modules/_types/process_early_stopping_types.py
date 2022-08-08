from typing import TypedDict, Literal



# Process Early Stopping
# Training and evaluating a model is slow and resource intensive. Therefore, 
# this module reviews the performance of a model and determines if the
# process should continue or be stopped.



# The names of the processes that make use of this module.
IEarlyStoppingProcessName = Literal["RegressionDiscovery", "ClassificationDiscovery", "ModelEvaluation"]




# Checkpoint
# A process can contain any number of checkpoints and they can be performed at
# any points in the dataset.


# Configuration passed through the constructor
class IProcessEarlyStoppingCheckpointConfig(TypedDict):
    # The minimum required positions
    required_longs: int
    required_shorts: int

    # The percentage of the dataset in which the checkpoint should be performed
    dataset_percent: float # This value should be a float ranging 0-1



# Checkpoint Dictionary built on class init
class IProcessEarlyStoppingCheckpoint(TypedDict):
    # The candlestick index in which the checkpoint should be evaluated
    index: int

    # The state of the checkpoint
    passed: bool

    # The minimum required positions
    required_longs: int
    required_shorts: int

    # The reason why the model's evaluation should be stopped early
    motive: str



