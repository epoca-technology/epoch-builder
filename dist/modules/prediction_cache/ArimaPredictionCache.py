from typing import Union, List
from modules.types import IPrediction
from modules.database.Database import Database





def save_arima_pred(
    model_id: str, 
    first_ot: int, 
    last_ct: int, 
    predictions: int, 
    interpreter_long: float, 
    interpreter_short: float, 
    pred: IPrediction
) -> None:
    """Saves an Arima Prediction in the Database.

    Args:
        model_id: str
            The ID of the model
        first_ot: int
            The open time of the first lookback candlestick.
        last_ct: int
            The close time of the last lookback candlestick.
        predictions: int
            The number of predictions the model outputs.
        interpreter_long: float
            The long arg set on the model's interpreter.
        interpreter_short: float
            The short arg set on the model's interpreter.
        pred: IPrediction
            The prediction to be stored in the db.
    """
    Database.write_query(
        f"\
            INSERT INTO {Database.tn('arima_predictions')}(id, fot, lct, pn, l, s, p) \
            VALUES (%s, %s, %s, %s, %s, %s, %s)\
        ",
        (model_id, first_ot, last_ct, predictions, interpreter_long, interpreter_short, pred)
    )






def get_arima_pred(
    model_id: str, 
    first_ot: int, 
    last_ct: int, 
    predictions: int, 
    interpreter_long: float, 
    interpreter_short: float, 
) -> Union[IPrediction, None]:
    """Retrieves a prediction from the db. If it is not found it returns None.

    Args:
        model_id: str
            The ID of the model
        first_ot: int
            The open time of the first lookback candlestick.
        last_ct: int
            The close time of the last lookback candlestick.
        predictions: int
            The number of predictions the model outputs.
        interpreter_long: float
            The long arg set on the model's interpreter.
        interpreter_short: float
            The short arg set on the model's interpreter.
    
    Returns:
        Union[IPrediction, None]
    """
    # Retrieve the prediction if any
    pred_snapshot: List[IPrediction] = Database.read_query(
        f"\
            SELECT p FROM {Database.tn('arima_predictions')} WHERE \
            id = %s AND fot = %s AND lct = %s AND pn = %s AND l = real %s AND s = real %s \
            LIMIT 1\
        ",
        (model_id, first_ot, last_ct, predictions, str(interpreter_long), str(interpreter_short))
    )

    # Return the prediction if any
    return pred_snapshot[0]['p'] if len(pred_snapshot) == 1 else None






def delete_arima_pred(
    model_id: str, 
    first_ot: int, 
    last_ct: int, 
    predictions: int, 
    interpreter_long: float, 
    interpreter_short: float, 
) -> None:
    """Deletes a prediction from the db. 

    Args:
        model_id: str
            The ID of the model
        first_ot: int
            The open time of the first lookback candlestick.
        last_ct: int
            The close time of the last lookback candlestick.
        predictions: int
            The number of predictions the model outputs.
        interpreter_long: float
            The long arg set on the model's interpreter.
        interpreter_short: float
            The short arg set on the model's interpreter.
    """
    Database.write_query(
        f"\
            DELETE FROM {Database.tn('arima_predictions')} WHERE \
            id = %s AND fot = %s AND lct = %s AND pn = %s AND l = real %s AND s = real %s\
        ",
        (model_id, first_ot, last_ct, predictions, str(interpreter_long), str(interpreter_short))
    )