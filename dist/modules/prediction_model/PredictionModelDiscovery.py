from typing import List
from modules._types import IDiscovery
from modules.discovery.Discovery import Discovery



class PredictionModelDiscovery(Discovery):
    """PredictionModelDiscovery Class

    This class builds the discovery for a Prediction Model.

    Instance Properties:
        ...
    """





    def __init__(self):
        """Initializes the RegressionDiscovery Instance.
        """
        # Initialize the Discovery Instance
        super().__init__(reward=1, penalty=1)










    def discover(self, features_sum: List[float], labels: List[float]) -> IDiscovery:
        """Executes the discover process for a Prediction Model.

        Args:
            features_sum: List[float]
                The list of feature sums.
            labels: List[float]
                The list of labels.

        Returns:
            IDiscovery
        """
        # Init values
        features_num: int = len(features_sum)
        labels_num: int = len(labels)

        # Iterate for as long there are features and labels
        i: int = 0
        while i < labels_num and i < features_num:
            # Handle an increase prediction
            if features_sum[i] > 0:
                self.increase_pred(pred=features_sum[i], successful=labels[i] == 1)

            # Handle a decrease prediction
            elif features_sum[i] < 0:
                self.decrease_pred(pred=features_sum[i], successful=labels[i] == -1)

            # Handle a neutral prediction
            else:
                self.neutral_pred()

            # Increase the counter
            i += 1

        # Finally, build and return the discovery
        return self.build()