from typing import Union, List
from modules.types import IPrediction
from modules.database.Database import Database
from modules.prediction_cache.Interface import PredictionCacheInterface




class ClassificationPredictionCache(PredictionCacheInterface):
    """ClassificationPredictionCache Class

    This class stores predictions generated by classification models in the database.

    Instance Properties:
        model_id: str
            The ID of the model that initialized the instance.
        min_probability: float
            The interpreter's configuration.
    """


    def __init__(self, model_id: str, min_probability: float):
        """Initializes the prediction cache instance.

        Args:
            model_id: str
                The ID of the model that initialized the instance.
            min_probability: float
                    The interpreter's configuration.
        """
        # Init the ID
        self.model_id: str = model_id

        # Init the interpreter's configuration
        self.min_probability: float = min_probability








    def get(self, first_ot: int, last_ct: int) -> Union[IPrediction, None]:
        """Retrieves a prediction. Returns None in case it doesn't exist.
        Make sure to always save predictions if they don't exist.

        Args:
            first_ot: int
            last_ct: int
                The prediction lookback range.

        Returns:
            Union[IPrediction, None]
        """
        # Retrieve the prediction if any
        pred_snapshot: List[IPrediction] = Database.read_query(
            f"\
                SELECT p FROM {Database.tn('regression_predictions')} WHERE \
                id = %s AND fot = %s AND lct = %s AND pn = %s AND l = real %s AND s = real %s \
                LIMIT 1\
            ",
            (self.model_id, first_ot, last_ct, self.predictions, str(self.interpreter_long), str(self.interpreter_short))
        )

        # Return the prediction if any
        return pred_snapshot[0]['p'] if len(pred_snapshot) == 1 else None







    def save(self, first_ot: int, last_ct: int, pred: IPrediction) -> None:
        """Saves a Regression Prediction in the Database.

        Args:
            first_ot: int
                The open time of the first lookback candlestick.
            last_ct: int
                The close time of the last lookback candlestick.
            pred: IPrediction
                The prediction to be stored in the db.
        """
        Database.write_query(
            f"\
                INSERT INTO {Database.tn('regression_predictions')}(id, fot, lct, pn, l, s, p) \
                VALUES (%s, %s, %s, %s, %s, %s, %s)\
            ",
            (self.model_id, first_ot, last_ct, self.predictions, self.interpreter_long, self.interpreter_short, pred)
        )








    def delete(self, first_ot: int, last_ct: int) -> None:
        """Deletes a prediction from the db. This functionality is only to be used on unit tests.

        Args:
            first_ot: int
                The open time of the first lookback candlestick.
            last_ct: int
                The close time of the last lookback candlestick.
        """
        Database.write_query(
            f"\
                DELETE FROM {Database.tn('regression_predictions')} WHERE \
                id = %s AND fot = %s AND lct = %s AND pn = %s AND l = real %s AND s = real %s\
            ",
            (self.model_id, first_ot, last_ct, self.predictions, str(self.interpreter_long), str(self.interpreter_short))
        )