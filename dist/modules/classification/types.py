from typing import TypedDict, List, Union, Dict
from modules.keras_models import IKerasModelConfig, IKerasModelTrainingHistory
from modules.model import IModel, IClassificationConfig












## CLASSIFICATION TRAINING DATA ##





# Active Training Data Position
# When a position is opened, the active position dict is populated with the up and down price values
# as well as the predictions generated by the Models. Once the position closes, the row dict
# is completed with the up and down values
class ITrainingDataActivePosition(TypedDict):
    up_price: float         # The price in which the position will be closed as up
    down_price: float       # The price in which the position will be closed as down
    row: Dict[str, float]   # Model's features which will be completed with labels once the position closes






# Training Data Insights
# Data used to represent the proportions for the price changes and the position types generated by the 
# models.
class ITrainingDataPriceActionsInsight(TypedDict):
    up: float
    down: float
class ITrainingDataPredictionInsight(TypedDict):
    long: float
    short: float
    neutral: float







# Training Data Config
# The Training configuration that resides in the configuration file and it is used to initialize
# the training data generator.
class ITrainingDataConfig(TypedDict):
    # The description of the Training Data that will be generated.
    description: str

    # Start and end time - If none provided, will use all the available data
    start: Union[str, int, None]
    end: Union[str, int, None]

    # Percentages that will determine if the price moved up or down after a position is opened
    up_percent_change: float
    down_percent_change: float

    # The list of ArimaModels|RegressionModels that will be used to predict
    models: List[IModel] # IModel does not exist yet





# Compressed Training Data
# In order to optimize the size of the training data file, the data is converted into a dict
# with the rows and columns lists.
class ICompressedTrainingData(TypedDict):
    columns: List[str]
    rows: List[List[float]]





# Training Data File
# The dict that contains all the information needed to train a ClassificationModel.
class ITrainingDataFile(TypedDict):
    # Universally Unique Identifier (uuid4)
    id: str

    # The description of the Training Data that will be generated.
    description: str

    # The timestamp in which the Training Data was generated
    creation: int

    # Start and end time
    start: int  # First candlestick's ot
    end: int    # Last candlestick's ct

    # The number of minutes that took to generate the training data
    duration_minutes: int

    # Percentages that will determine if the price moved up or down after a position is opened
    up_percent_change: float
    down_percent_change: float

    # List of ArimaModels|RegressionModels
    models: List[IModel] # IModel does not exist yet

    # Price Actions Insight - The up and down total count
    price_actions_insight: ITrainingDataPriceActionsInsight

    # Prediction Insight 
    # Position type count for each ArimaModel|RegressionModel in this format:
    # {[modelID: str]: ITrainingDataPredictionInsight}
    predictions_insight: Dict[str, Dict[str, float]]

    # Training Data
    # The training data generated in a compressed format.
    training_data: ICompressedTrainingData











## CLASSIFICATION TRAINING ##




# Classification Training Configuration
# The configuration that will be used to initialize, train and save the models.
class IClassificationTrainingConfig(TypedDict):
    # The ID of the model. Must be descriptive, compatible with filesystems and preffixed with 'C_'
    id: str

    # Any relevant data that should be attached to the trained model.
    description: str

    # The learning rate to be used by the optimizer
    learning_rate: float

    # The optimizer to be used.
    optimizer: str # 'adam'|'rmsprop'

    # The loss function to be used
    loss: str # 'categorical_crossentropy'|'?'

    # The metric to be used for meassuring the val_loss
    metric: str # 'categorical_accuracy'|'?'

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
    # Identifier
    id: str
    description: str

    # Date Range
    start: int    # Open Time of the first prediction candlestick
    end: int      # Close Time of the last prediction candlestick

    # Dataset Sizes
    train_size: int     # Number of rows in the train dataset
    test_size: int      # Number of rows in the test dataset

    # Percentages that determine if the price moved up or down
    up_percent_change: float
    down_percent_change: float






# Classification Evaluation
# Evaluation performed right after the model is trained in order to get an overview of the
# potential accuracy, as well as the prediction type distribution.
# Each evaluation is performed using a random candlestick open time and is evaluated against
# the sequence of 1 minute candlesticks that follow. The iteration will continue until the
# evaluation position is closed.
class IClassificationEvaluation(TypedDict):
    # The number of evaluations performed on the Regression
    evaluations: int
    max_evaluations: int

    # The number of times the Regression predicted a price increase
    increase_num: int
    increase_successful_num: int

    # The number of times the Regression predicted a price decrease
    decrease_num: int
    decrease_successful_num: int

    # Accuracy
    increase_acc: int
    decrease_acc: int
    acc: int

    # Increase Predictions Overview
    increase_list: List[float]
    increase_max: float
    increase_min: float
    increase_mean: float
    increase_successful_max: float
    increase_successful_min: float
    increase_successful_mean: float

    # Decrease Predictions Overview
    decrease_list: List[float]
    decrease_max: float
    decrease_min: float
    decrease_mean: float
    decrease_successful_max: float
    decrease_successful_min: float
    decrease_successful_mean: float

    # Outcomes
    increase_outcomes: int
    decrease_outcomes: int








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
    learning_rate: float
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
    classification_evaluation: IClassificationEvaluation

    # The configuration of the Classification
    classification_config: IClassificationConfig