from typing import TypedDict, List
from modules.types import IKerasModelConfig, IKerasModelTrainingHistory, IClassificationConfig,\
    IModelEvaluation







## CLASSIFICATION TRAINING ##




# Classification Training Configuration
# The configuration that will be used to initialize, train and save the models.
class IClassificationTrainingConfig(TypedDict):
    # The ID of the model. Must be descriptive, compatible with filesystems and preffixed with 'C_'
    id: str

    # Any relevant data that should be attached to the trained model.
    description: str

    # The optimizer to be used.
    optimizer: str # 'adam'|'rmsprop'

    # The loss function to be used
    loss: str # 'categorical_crossentropy'|'binary_crossentropy'

    # The metric to be used for meassuring the val_acc
    metric: str # 'categorical_accuracy'|'binary_accuracy'

    # Keras Model Configuration
    keras_model: IKerasModelConfig







# Classification Training Batch
# Keras Models and created and evaluated in batches. Moreover, multiple batches can be combined
# in the GUI
class IClassificationTrainingBatch(TypedDict):
    # Descriptive name to easily identify the batch. Must be compatible with filesystems.
    name: str

    # ID of the Classification Training Data that will be used to train all the models.
    training_data_id: str

    # The configurations for the models that will be trained within the batch.
    models: List[IClassificationTrainingConfig]







# Training Data Summary
# In order to simplify interactions with the IClassificationTrainingCertificate, the training
# data is summarized in a dictionary.
class ITrainingDataSummary(TypedDict):
    # The ID of the Regression Selection that was used to pick the Regression Models
    regression_selection_id: str

    # Identifier
    id: str
    description: str

    # Date Range
    start: int    # Open Time of the first prediction candlestick
    end: int      # Close Time of the last prediction candlestick

    # Dataset Sizes
    train_size: int     # Number of rows in the train dataset
    test_size: int      # Number of rows in the test dataset

    # The Prediction Candlestick steps that will be used to generate the data. If 0 is provided
    # the training data will be generated the traditional way.
    # The purpose of this mode is to increase the size of the Training Dataset and cover more 
    # cases.
    steps: int

    # Percentages that determine if the price moved up or down
    up_percent_change: float
    down_percent_change: float

    # Optional Technical Analysis Features
    include_rsi: bool       # Momentum
    include_stoch: bool     # Momentum
    include_aroon: bool     # Trend
    include_stc: bool       # Trend
    include_mfi: bool       # Volume

    # The total number of features that will be used by the model to predict
    features_num: int













# Classification Training Certificate
# Once the training, saving and evaluation completes, a certificate containing all the
# data is saved and issued for batching.
class IClassificationTrainingCertificate(TypedDict):
    ## Identification ##
    id: str
    description: str


    ## Training Data ##
    training_data_summary: ITrainingDataSummary



    ## Training Configuration ##
    optimizer: str
    loss: str
    metric: str
    keras_model_config: IKerasModelConfig




    ## Training ##

    # Date Range
    training_start: int     # Time in which the training started
    training_end: int       # Time in which the training ended

    # Training performance by epoch
    training_history: IKerasModelTrainingHistory

    # Result of the evaluation of the test dataset
    test_evaluation: List[float] # [loss, metric]

    # Classification Post-Training Evaluation
    classification_evaluation: IModelEvaluation

    # The configuration of the Classification
    classification_config: IClassificationConfig