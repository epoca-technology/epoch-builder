from typing import List
from numpy import mean, median
from modules._types import IDiscovery
from modules.utils.Utils import Utils




class Discovery:
    """Discovery Class

    This class implements the model discovery process and can be extended by any
    type of model.

    Instance Properties:
        Points:
            reward: float
                The number of points that will be added when predicting correctly.
            penalty: float
                The number of points that will be deducted when predicting incorrectly.
            points_hist: List[float]
                The history of the collected points.

        Predictions:
            neutral_num: int
                The number of neutral predictions generated.
            increase: List[float]
                The list of all increase predictions.
            increase_successful: List[float]
            increase_unsuccessful: List[float]
                The list of all successful & unsuccessful increase predictions.
            decrease: List[float]
                The list of all decrease predictions.
            decrease_successful: List[float]
            decrease_unsuccessful: List[float]
                The list of all successful & unsuccessful decrease predictions.

        Outcomes:
            neutral_outcome_num: int
            increase_outcome_num: int
            decrease_outcome_num: int
                The number of outcomes by type.
    """





    def __init__(self):
        """Initializes the Discovery Instance.
        """
        # Init the point values
        self.reward: float = 1
        self.penalty: float = 1.3
        self.points_hist: List[float] = [0]

        # Init the neutral prediction values
        self.neutral_num: int = 0

        # Init the increase prediction values
        self.increase: List[float] = []
        self.increase_successful: List[float] = []
        self.increase_unsuccessful: List[float] = []

        # Init the decrease prediction values
        self.decrease: List[float] = []
        self.decrease_successful: List[float] = []
        self.decrease_unsuccessful: List[float] = []

        # Init the outcome values
        self.neutral_outcome_num: int = 0
        self.increase_outcome_num: int = 0
        self.decrease_outcome_num: int = 0












    #######################
    ## Discovery Process ##
    #######################




    def discover(self, *args, **kwargs) -> IDiscovery:
        raise NotImplementedError("Discovery.discover has not been implemented correctly.")















    ######################
    ## Discovery Events ##
    ######################





    def neutral_pred(self) -> None:
        """Increases the neutral prediction counter.
        """
        self.neutral_num += 1







    def neutral_outcome(self) -> None:
        """Increases the neutral outcome counter.
        """
        self.neutral_outcome_num += 1







    def increase_pred(self, pred: float, successful: bool) -> None:
        """Handles an increase prediction event based on its outcome.

        Args:
            pred: float
                The generated prediction.
            successful: bool
                The result of the prediction
        """
        # Add the prediction to the list
        self.increase.append(pred)

        # Handle a successful increase prediction
        if successful:
            self.increase_successful.append(pred)
            self.points_hist.append(self.points_hist[-1] + self.reward)
            self.increase_outcome_num += 1

        # Handle an unsuccessful increase prediction
        else:
            self.increase_unsuccessful.append(pred)
            self.points_hist.append(self.points_hist[-1] - self.penalty)
            self.decrease_outcome_num += 1







    def decrease_pred(self, pred: float, successful: bool) -> None:
        """Handles a decrease prediction event based on its outcome.

        Args:
            pred: float
                The generated prediction.
            successful: bool
                The result of the prediction
        """
        # Add the prediction to the list
        self.decrease.append(pred)

        # Handle a successful decrease prediction
        if successful:
            self.decrease_successful.append(pred)
            self.points_hist.append(self.points_hist[-1] + self.reward)
            self.decrease_outcome_num += 1

        # Handle an unsuccessful decrease prediction
        else:
            self.decrease_unsuccessful.append(pred)
            self.points_hist.append(self.points_hist[-1] - self.penalty)
            self.increase_outcome_num += 1



















    #######################
    ## Discovery Builder ##
    #######################




    def build(self) -> IDiscovery:
        """Builds the discovery and the payload based on the collected
        data.

        Returns:
            IDiscovery
        """
        # Init values
        increase_num: int = len(self.increase)
        increase_successful_num = len(self.increase_successful)
        increase_unsuccessful_num = len(self.increase_unsuccessful)
        decrease_num: int = len(self.decrease)
        decrease_successful_num = len(self.decrease_successful)
        decrease_unsuccessful_num = len(self.decrease_unsuccessful)
        total_num = increase_num + decrease_num
        total_successful_num = increase_successful_num + decrease_successful_num

        # Build the discovery
        return {
            # Predictions generated by the model
            "neutral_num": self.neutral_num,
            "increase_num": increase_num,
            "decrease_num": decrease_num,

            # Outcomes during the discovery
            "neutral_outcome_num": self.neutral_outcome_num,
            "increase_outcome_num": self.increase_outcome_num,
            "decrease_outcome_num": self.decrease_outcome_num,

            # The points collected during the discovery
            "points_hist": self.points_hist,
            "points": round(self.points_hist[-1], 2),

            # The accuracy of the generated predictions
            "increase_accuracy": self._calculate_accuracy(increase_successful_num, increase_num),
            "decrease_accuracy": self._calculate_accuracy(decrease_successful_num, decrease_num),
            "accuracy": self._calculate_accuracy(total_successful_num, total_num),

            # Details of the increase predictions
            "increase_list": self.increase,
            "increase_min": self._calculate_min(self.increase, increase_num),
            "increase_max": self._calculate_max(self.increase, increase_num),
            "increase_mean": self._calculate_mean(self.increase, increase_num),
            "increase_median": self._calculate_median(self.increase, increase_num),

            # Details of the decrease predictions
            "decrease_list": self.decrease,
            "decrease_min": self._calculate_min(self.decrease, decrease_num),
            "decrease_max": self._calculate_max(self.decrease, decrease_num),
            "decrease_mean": self._calculate_mean(self.decrease, decrease_num),
            "decrease_median": self._calculate_median(self.decrease, decrease_num),

            # Details of the successful increase predictions
            "increase_successful_list": self.increase_successful,
            "increase_successful_mean": self._calculate_mean(self.increase_successful, increase_successful_num),
            "increase_successful_median": self._calculate_median(self.increase_successful, increase_successful_num),

            # Details of the unsuccessful increase predictions
            "increase_unsuccessful_list": self.increase_unsuccessful,
            "increase_unsuccessful_mean": self._calculate_mean(self.increase_unsuccessful, increase_unsuccessful_num),
            "increase_unsuccessful_median": self._calculate_median(self.increase_unsuccessful, increase_unsuccessful_num),

            # Details of the successful decrease predictions
            "decrease_successful_list": self.decrease_successful,
            "decrease_successful_mean": self._calculate_mean(self.decrease_successful, decrease_successful_num),
            "decrease_successful_median": self._calculate_median(self.decrease_successful, decrease_successful_num),

            # Details of the unsuccessful decrease predictions
            "decrease_unsuccessful_list": self.decrease_unsuccessful,
            "decrease_unsuccessful_mean": self._calculate_mean(self.decrease_unsuccessful, decrease_unsuccessful_num),
            "decrease_unsuccessful_median": self._calculate_median(self.decrease_unsuccessful, decrease_unsuccessful_num)
        }








    def _calculate_accuracy(self, value: int, total: int) -> float:
        """Calculates the accuracy received during the discovery.

        Args:
            value: int
                The number of accurate predictions generated by the model.
            total: int
                The total number of predictions generated by the model.

        Returns:
            float
        """
        return round(Utils.get_percentage_out_of_total(value, total if total > 0 else 1), 2)






    def _calculate_min(self, values: List[float], total: int) -> float:
        """Calculates the minimum value within a list of floats.

        Args:
            values: List[float]
                The list of values it will calculate the min for.
            total: int
                The total number of values in the list. If there are not 
                items, it will perform the calculation safely.

        Returns:
            float
        """
        return round(min(values if total > 0 else [0]), 6)





    def _calculate_max(self, values: List[float], total: int) -> float:
        """Calculates the maximum value within a list of floats.

        Args:
            values: List[float]
                The list of values it will calculate the min for.
            total: int
                The total number of values in the list. If there are not 
                items, it will perform the calculation safely.

        Returns:
            float
        """
        return round(max(values if total > 0 else [0]), 6)





    def _calculate_mean(self, values: List[float], total: int) -> float:
        """Calculates the mean value within a list of floats.

        Args:
            values: List[float]
                The list of values it will calculate the min for.
            total: int
                The total number of values in the list. If there are not 
                items, it will perform the calculation safely.

        Returns:
            float
        """
        return round(mean(values if total > 0 else [0]), 6)





    def _calculate_median(self, values: List[float], total: int) -> float:
        """Calculates the median value within a list of floats.

        Args:
            values: List[float]
                The list of values it will calculate the min for.
            total: int
                The total number of values in the list. If there are not 
                items, it will perform the calculation safely.

        Returns:
            float
        """
        return round(median(values if total > 0 else [0]), 6)