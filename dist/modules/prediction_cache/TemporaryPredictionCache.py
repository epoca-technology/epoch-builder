from typing import Dict, Union
from modules.model import IPrediction






class TemporaryPredictionCache:
    """TemporaryPredictionCache Class

    This class stores predictions temporarily in a key value basis per Model Instance.

    Instance Properties:
        predictions: Dict[str, IPrediction]
            The dict that stores the predictions based on the lookback prediction range.
    """


    ## Initialization ##
    def __init__(self):
        """Initializes the temporary cache instance.
        """
        self.predictions: Dict[str, IPrediction] = {}







    ## Management ##




    def get(self, first_ot: int, last_ct: int) -> Union[IPrediction, None]:
        """Retrieves a prediction if exists. Otherwise, save it afterwards.

        Args:
            first_ot: int
            last_ct: int

        Returns:
            Union[IPrediction, None]
        """
        return self.predictions.get(self._get_id(first_ot, last_ct))






    def save(self, first_ot: int, last_ct: int, pred: IPrediction) -> None:
        """Saves a prediction based on the prediction range into the cache.

        Args:
            first_ot: int 
            last_ct: int 
            pred: IPrediction
                The prediction to be stored in the temp memory.
        """
        self.predictions[self._get_id(first_ot, last_ct)] = pred







    ## Helpers ##


    def _get_id(self, first_ot: int, last_ct: int) -> str:
        """Generates an ID based on the prediction range.

        Args:
            first_ot: int
            first_ot: int

        Returns:
            str
        """
        return f"{str(int(first_ot))}_{str(int(last_ct))}"
