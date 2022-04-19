from typing import List
from collections import Counter
from modules.utils import Utils
from modules.model import SingleModel, IModel, IPrediction, IPredictionMetaData



class MultiModel:
    """Multi Model Class

    Initializes a MultiModel's instance that is ready to perform predictions.

    Instance Properties:
        id: string
            The name of the MultiModel.
        consensus: int
            The number of models that must agree to predict a long or a short.
        single_models: List[SingleModel]
            The list of models that will be used to perform predictions.
    """


    def __init__(self, config: IModel):
        """Initializes the instance properties as well as the models.

        Args:
            config: IModel
                The configuration dict that will be used to initialize the instance.
        
        Raises:
            ValueError:
                If less than 2 single models are passed.
                If the consensus is less than 1 or greater than the number of models passed.
                If the consensus doesnt represent more than 50% of the models.

        """
        # Make sure the list of configs has at least 2 items
        if len(config["single_models"]) < 2:
            raise ValueError(f"A MultiModel can only be initialized with 2 or more single models. Received {len(config['single_models'])}")

        # Make sure the consensus is valid if it was provided
        if isinstance(config.get('consensus'), int) and (config.get('consensus') < 1 or config.get('consensus') > len(config['single_models'])):
            raise ValueError(f"The consensus cannot be smaller than 1 or greater than the number of single models. \
                Received: {config.get('consensus')}")

        # Init the basic properties
        self.id = config['id']
        self.consensus = config['consensus'] if isinstance(config.get('consensus'), int) else len(config['single_models'])

        # The consensus must represent more than 50% of the models quantity
        if Utils.get_percentage_out_of_total(self.consensus, len(config['single_models'])) <= 50:
            raise ValueError(f"The consensus must represent more than 50% of the single models. \
                Received: {self.consensus} / {len(config['single_models'])}")

        # Init the single models
        self.single_models = [SingleModel({'id': self.id, 'single_models': [m]}) for m in config['single_models']]







    def predict(self, current_timestamp: int) -> IPrediction:
        """Given the current time, it will perform a prediction on each model,
        evaluate the results and return them along with the metadata.

        Args:
            current_timestamp: int
                The current time in milliseconds.

        Returns:
            IPrediction
        """
        # Perform the Predictions
        results: List[int] = []
        metadata: List[IPredictionMetaData] = []
        for pred in [m.predict(current_timestamp) for m in self.single_models]:
            results.append(pred['r'])
            metadata.append(pred['md'][0])
        
        # Return the Prediction Results
        return {
            'r': self._get_prediction_result(results),
            't': current_timestamp,
            'md': metadata
        }







    def _get_prediction_result(self, results: List[int]) -> int:
        """Based on a list of results and the model's consensus, it will determine
        the final result of the prediction.

        Args:
            results: List[int]
                The list of results predicted by all models.

        Returns:
            int (-1, 0, 1)
        """
        # Create the Counter Instance
        counter: Counter = Counter(results)

        # Check if there is consensus for a long prediction
        if counter[1] >= self.consensus:
            return 1

        # Check if there is consensus for a short prediction
        elif counter[-1] >= self.consensus:
            return -1

        # Otherwise, return neutral
        else:
            return 0










    def get_max_lookback(self) -> int:
        """Returns the max lookback value in all models.

        Args:
            None

        Returns:
            int
        """
        return max([m.lookback for m in self.single_models])








    def get_model(self) -> IModel:
        """Dumps the MultiModel's data into a dictionary that will be used
        to get the insights based on its performance.

        Args:
            None

        Returns:
            IModel
        """
        return {
            "id": self.id,
            "consensus": self.consensus,
            "single_models": [m.get_model()['single_models'][0] for m in self.single_models]
        }