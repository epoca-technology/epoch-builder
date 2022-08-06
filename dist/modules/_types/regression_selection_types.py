from typing import TypedDict, List, Dict
from modules._types.model_types import IModel







# Regression Selection File
# The result of a RegressionSelection that contains all the information related
# to the selection and the models in it.
class IRegressionSelectionFile(TypedDict):
    # Universally Unique Identifier (uuid4)
    id: str
    
    # The number of models that were selected based on their points medians
    models_limit: int

    # The data range covered in the backtest results
    start: int
    end: int

    # The number of models that were put through the selection
    models_num: int

    # The list of ordered combination results 
    results: List[Dict]



