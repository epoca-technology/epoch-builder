import os
from typing import Union
from sqlitedict import SqliteDict
from modules.model import IPrediction



# If the Database's directory doesn't exist, create it
DB_PATH: str = 'db'
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)


## Database Init ##
DB: SqliteDict = SqliteDict(f"{DB_PATH}/db.sqlite", tablename="arima_predictions", autocommit=True, outer_stack=False)





# Predictions Management
# In order to accelerate the execution times of the backtesting, predictions are saved in a 
# local database in a key: val format based on the id of the model, the first ot and the last
# ct of the lookback range.
# An example of a key that holds a prediction is: A601_1502942400000_1509139799999
# The keys may be longer in Regression and Classification Models. So far, the limit of the keys
# is unknown. However, they impact performance.




def save_pred(model_id: str, first_ot: int, last_ct: int, pred: IPrediction) -> None:
    """Saves a Model Prediction in the database for optimization purposes.

    Args:
        model_id: str
            The ID of the Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.
        pred: IPrediction
            The prediction to save in the db.
    """
    DB[_get_pred_key(model_id, first_ot, last_ct)] = pred





def get_pred(model_id: str, first_ot: int, last_ct: int) -> Union[IPrediction, None]:
    """Retrieves a Model Prediction if it exists, otherwise returns None.

    Args:
        model_id: str
            The ID of the Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.

    Returns:
        Union[IPrediction, None]
    """
    return DB.get(_get_pred_key(model_id, first_ot, last_ct))





def delete_pred(model_id: str, first_ot: int, last_ct: int) -> None:
    """Deletes a Model Prediction from the Database.

    Args:
        model_id: str
            The ID of the Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.
    """
    # Init the key
    key: str = _get_pred_key(model_id, first_ot, last_ct)

    # if the record exists, delete it
    if key in DB:
        del DB[key]




def _get_pred_key(model_id: str, first_ot: int, last_ct: int) -> str:
    """Returns the key that should be used to save or retrieve the prediction.

    Args:
        model_id: str
            The ID of the Arima Model. F.e: A601
        first_ot: int
            The open timestamp of the first prediction candlestick.
        last_ct: int
            The close timestamp of the last prediction candlestick.
    
    Returns:
        str
    """
    return f"{model_id}_{first_ot}_{last_ct}"