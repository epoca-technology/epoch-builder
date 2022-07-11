from typing import TypedDict, List, Union, Dict
from modules.types import IModel








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
    # The ID of the Regression Selection that was used to pick the Regression Models
    regression_selection_id: str

    # The description of the Training Data that will be generated.
    description: str

    # Start and end time - If none provided, will use all the available data
    start: Union[str, int, None]
    end: Union[str, int, None]

    # The Prediction Candlestick steps that will be used to generate the data. If 0 is provided
    # the training data will be generated the traditional way.
    # The purpose of this mode is to increase the size of the Training Dataset and cover more 
    # cases.
    steps: int

    # Percentages that will determine if the price moved up or down after a position is opened
    up_percent_change: float
    down_percent_change: float

    # The list of ArimaModels|RegressionModels that will be used to predict
    models: List[IModel]

    # Optional Technical Analysis Features
    include_rsi: bool       # Momentum
    include_stoch: bool     # Momentum
    include_aroon: bool     # Trend
    include_stc: bool       # Trend
    include_mfi: bool       # Volume





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

    # The ID of the Regression Selection that was used to pick the Regression Models
    regression_selection_id: str

    # The description of the Training Data that will be generated.
    description: str

    # The timestamp in which the Training Data was generated
    creation: int

    # Start and end time
    start: int  # First candlestick's ot
    end: int    # Last candlestick's ct

    # The number of minutes that took to generate the training data
    duration_minutes: int

    # The Prediction Candlestick steps that will be used to generate the data. If 0 is provided
    # the training data will be generated the traditional way.
    # The purpose of this mode is to increase the size of the Training Dataset and cover more 
    # cases.
    steps: int

    # Percentages that will determine if the price moved up or down after a position is opened
    up_percent_change: float
    down_percent_change: float

    # List of ArimaModels|RegressionModels
    models: List[IModel]

    # Optional Technical Analysis Features
    include_rsi: bool       # Momentum
    include_stoch: bool     # Momentum
    include_aroon: bool     # Trend
    include_stc: bool       # Trend
    include_mfi: bool       # Volume

    # The total number of features that will be used by the model to predict
    features_num: int

    # Price Actions Insight - The up and down total count
    price_actions_insight: ITrainingDataPriceActionsInsight

    # Prediction Insight 
    # Position type count for each ArimaModel|RegressionModel in this format:
    # {[modelID: str]: ITrainingDataPredictionInsight}
    predictions_insight: Dict[str, Dict[str, float]]

    # Technical Analysis Summary
    # If none of the technical analysis features are enabled, this value will be None.
    # {[taName: str]: df.describe().to_dict()}|null
    technical_analysis_insight: Union[Dict[str, Dict[str, float]], None]

    # Training Data
    # The training data generated in a compressed format.
    training_data: ICompressedTrainingData


