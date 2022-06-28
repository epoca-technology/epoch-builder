from typing import Union
from modules.types import IPrediction



# Prediction Cache Interface
# RegressionPredictionCache & ClassificationPredictionCache implement the following interface
# in order to ensure compatibility across any of the processes.
class PredictionCacheInterface:
    # Retrieves a prediction. Returns None in case it doesn't exist.
    # Make sure to always save predictions if they don't exist.
    def get(self, first_ot: int, last_ct: int) -> Union[IPrediction, None]:
        raise NotImplementedError("PredictionCache.get has not been implemented.")


    # Saves a Prediction in the Database.
    def save(self, first_ot: int, last_ct: int, pred: IPrediction) -> None:
        raise NotImplementedError("PredictionCache.save has not been implemented.")


    # Deletes a prediction from the db. This functionality is only to be used on unit tests.
    def delete(self, first_ot: int, last_ct: int) -> None:
        raise NotImplementedError("PredictionCache.delete has not been implemented.")






