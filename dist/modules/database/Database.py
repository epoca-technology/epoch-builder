import os
from typing import Union
from sqlitedict import SqliteDict
from modules.model import IPrediction



# If the Database's directory doesn't exist, create it
if not os.path.exists('./database'):
    os.makedirs('./database')


## Databases Init ##
ARIMA_PREDICTIONS_DB = SqliteDict("db/db.sqlite", tablename="arima_predictions", autocommit=True, outer_stack=False)








## Arima Predictions ##




def save_prediction(model_id: str, first_ot: int, last_ct: int, pred: IPrediction) -> None:
    """Saves an Arima prediction in the database for optimization purposes.

    Args:
        model_id: str
            The ID of the Arima Model. F.e: A601
        first_ot: Union[int, float]
            The open timestamp of the first prediction candlestick.
        last_ct: Union[int, float]
            The close timestamp of the last prediction candlestick.
        pred: IPrediction
            The prediction to save in the db.
    """
    ARIMA_PREDICTIONS_DB[_get_prediction_key(model_id, first_ot, last_ct)] = pred




def get_prediction(model_id: str, first_ot: Union[int, float], last_ct: Union[int, float]) -> Union[IPrediction, None]:
    """Retrieves an arima prediction if it currently exists, otherwise returns None. If a prediction is found,
    the 't' property will be replaced with the provided current_timestamp.

    Args:
        model_id: str
            The ID of the Arima Model. F.e: A601
        first_ot: Union[int, float]
            The open timestamp of the first prediction candlestick.
        last_ct: Union[int, float]
            The close timestamp of the last prediction candlestick.

    Returns:
        Union[IPrediction, None]
    """
    return ARIMA_PREDICTIONS_DB.get(_get_prediction_key(model_id, first_ot, last_ct))





def delete_prediction(model_id: str, first_ot: Union[int, float], last_ct: Union[int, float]) -> None:
    """Deletes an Arima Prediction from the Database.

    Args:
        model_id: str
            The ID of the Arima Model. F.e: A601
        first_ot: Union[int, float]
            The open timestamp of the first prediction candlestick.
        last_ct: Union[int, float]
            The close timestamp of the last prediction candlestick.

    Returns:
        Union[IPrediction, None]
    """
    # Init the key
    key: str = _get_prediction_key(model_id, first_ot, last_ct)

    # if the record exists, delete it
    if key in ARIMA_PREDICTIONS_DB:
        del ARIMA_PREDICTIONS_DB[key]




def _get_prediction_key(model_id: str, first_ot: Union[int, float], last_ct: Union[int, float]) -> str:
    """Returns the key that should be used to save or retrieve the prediction.

    Args:
        model_id: str
            The ID of the Arima Model. F.e: A601
        first_ot: Union[int, float]
            The open timestamp of the first prediction candlestick.
        last_ct: Union[int, float]
            The close timestamp of the last prediction candlestick.
    
    Returns:
        str
    """
    return f"{model_id}_{first_ot}_{last_ct}"