from typing import List
from numpy import ndarray
from modules._types import IDiscovery
from modules.utils.Utils import Utils
from modules.discovery.Discovery import Discovery



class RegressionDiscovery(Discovery):
    """RegressionDiscovery Class

    This class builds the discovery for a Regression Model.

    Instance Properties:
        ...
    """





    def __init__(self):
        """Initializes the RegressionDiscovery Instance.
        """
        # Initialize the Discovery Instance
        super().__init__()










    def discover(self, features: ndarray, labels: ndarray, preds: List[List[float]]) -> IDiscovery:
        """Executes the discover process for a Regression.

        Args:
            features: ndarray
                Test dataset features.
            labels: ndarray
                Test dataset labels.
            preds: List[List[float]]
                Test dataset predictions.

        Returns:
            IDiscovery
        """
        # Iterate over each prediction
        for i, pred in enumerate(preds):
            # Calculate the real change in price
            real_change: float = Utils.get_percentage_change(features[i, -1], labels[i, -1])

            # Calculate the predicted change in price
            predicted_change: float = Utils.get_percentage_change(features[i, -1], pred[-1])

            # Make sure there is an outcome
            if real_change != 0:
                # Handle an increase prediction
                if predicted_change > 0:
                    self.increase_pred(pred=predicted_change, successful=real_change > 0)

                # Handle a decrease prediction
                elif predicted_change < 0:
                    self.decrease_pred(pred=predicted_change, successful=real_change < 0)

                # Handle prediction neutrality
                else:
                    self.neutral_pred()

            # Otherwise, handle the outcome neutrality
            else:
                self.neutral_outcome()

        # Finally, build and return the discovery
        return self.build()



