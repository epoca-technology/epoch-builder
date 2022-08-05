from typing import TypedDict, List
from modules.types.model_types import IXGBClassificationConfig
from modules.types.classification_training_data_types import ITrainingDataSummary
from modules.types.model_evaluation_types import IModelEvaluation







## CLASSIFICATION TRAINING ##




# XGBoost Classification Training Configuration
# The configuration that will be used to initialize, train and save the models.
class IXGBClassificationTrainingConfig(TypedDict):
    # The ID of the model.
    id: str

    # Any relevant data that should be attached to the trained model.
    description: str

    # @TODO







# XGBoost Classification Training Batch
# XGB Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI
class IXGBClassificationTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # ID of the Classification Training Data that will be used to train all the models.
    training_data_id: str

    # The configurations for the models that will be trained within the batch.
    models: List[IXGBClassificationTrainingConfig]














# Classification Training Certificate
# Once the training, saving and evaluation completes, a certificate containing all the
# data is saved and issued for batching.
class IXGBClassificationTrainingCertificate(TypedDict):
    # Identification
    id: str
    description: str

    # Training Data
    training_data_summary: ITrainingDataSummary

    # Training Configuration
    # @TODO

    # Date Range
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    # @TODO

    # Result of the evaluation of the test dataset
    # @TODO

    # Classification Post-Training Evaluation
    classification_evaluation: IModelEvaluation

    # The configuration of the Classification
    classification_config: IXGBClassificationConfig