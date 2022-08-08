from typing import TypedDict, List
from modules._types.model_types import IKerasClassificationConfig
from modules._types.keras_models_types import IKerasModelConfig, IKerasModelTrainingHistory, \
    IKerasOptimizer, IKerasClassificationLoss, IKerasClassificationMetric, IKerasOptimizerName
from modules._types.classification_training_data_types import ITrainingDataSummary
from modules._types.discovery_types import IDiscoveryPayload
from modules._types.model_evaluation_types import IModelEvaluation







## CLASSIFICATION TRAINING ##




# Keras Classification Training Configuration
# The configuration that will be used to initialize, train and save the models.
class IKerasClassificationTrainingConfig(TypedDict):
    # The ID of the model.
    id: str

    # Any relevant data that should be attached to the trained model.
    description: str

    # The optimizer to be used.
    optimizer: IKerasOptimizer # 'adam'|'rmsprop'

    # The loss function to be used
    loss: IKerasClassificationLoss # 'categorical_crossentropy'|'binary_crossentropy'

    # The metric to be used for meassuring the val_acc
    metric: IKerasClassificationMetric # 'categorical_accuracy'|'binary_accuracy'

    # Keras Model Configuration
    keras_model: IKerasModelConfig







# Keras Classification Training Batch
# Keras Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI
class IKerasClassificationTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # ID of the Classification Training Data that will be used to train all the models.
    training_data_id: str

    # The configurations for the models that will be trained within the batch.
    models: List[IKerasClassificationTrainingConfig]














# Classification Training Certificate
# Once the training, saving and evaluation completes, a certificate containing all the
# data is saved and issued for batching.
class IKerasClassificationTrainingCertificate(TypedDict):
    # Identification
    id: str
    description: str

    # Training Data
    training_data_summary: ITrainingDataSummary

    # Training Configuration
    optimizer: IKerasOptimizerName
    loss: IKerasClassificationLoss
    metric: IKerasClassificationMetric
    keras_model_config: IKerasModelConfig

    # Date Range
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    training_history: IKerasModelTrainingHistory

    # Result of the evaluation of the test dataset
    test_evaluation: List[float] # [loss, metric]

    # Classification Discovery
    discovery: IDiscoveryPayload

    # Classification Post-Training Evaluation
    classification_evaluation: IModelEvaluation

    # The configuration of the Classification
    classification_config: IKerasClassificationConfig