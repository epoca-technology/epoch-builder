from typing import List, Tuple
from modules.types import IProbabilityInterpreterConfig, IPredictionResult
from modules.interpreter.Interface import InterpreterInterface




class ProbabilityInterpreter(InterpreterInterface):
    """ProbabilityInterpreter Class

    This class takes the up and down probabilities and performs an interpretation.

    Class Properties:
        MIN_PROBABILITY_VALUE: float
        MIN_PROBABILITY_VALUE: float
            The minimum and maximum values allowed for the min increase and decrease probabilities.

    Instance Properties:
        min_increase_probability: float
        min_decrease_probability: float
            The minimum probability up or down must have in order to return a non-neutral
            prediction result. Anything under the provided probability will be considered
            neutral.
    """

    # Minimum percentage change value for long and short configuration
    MIN_PROBABILITY_VALUE: float = 0.51
    MAX_PROBABILITY_VALUE: float = 0.99

  




    def __init__(self, config: IProbabilityInterpreterConfig):
        """Initializes the Interpreter Class based on the provided configuration.

        Args:
            config: IPercentChangeInterpreterConfig
                The configuration to interpret predictions.

        Raises:
            ValueError:
                If the min_increase_probability or min_decrease_probability is invalid.
        """
        # Validate the min_increase_probability
        if not isinstance(config.get("min_increase_probability"), float) or\
            config["min_increase_probability"] < ProbabilityInterpreter.MIN_PROBABILITY_VALUE or\
                config["min_increase_probability"] > ProbabilityInterpreter.MAX_PROBABILITY_VALUE:
            raise ValueError(f"The provided min_increase_probability is invalid: {config['min_increase_probability']}")

        # Validate the min_decrease_probability
        if not isinstance(config.get("min_decrease_probability"), float) or\
            config["min_decrease_probability"] < ProbabilityInterpreter.MIN_PROBABILITY_VALUE or\
                config["min_decrease_probability"] > ProbabilityInterpreter.MAX_PROBABILITY_VALUE:
            raise ValueError(f"The provided min_decrease_probability is invalid: {config['min_decrease_probability']}")

        # Initialize the instance properties
        self.min_increase_probability: float = config["min_increase_probability"]
        self.min_decrease_probability: float = config["min_decrease_probability"]









 



    def interpret(self, probabilities: List[float]) -> Tuple[IPredictionResult, str]:
        """Given a list of probabilities for up and down, it will check if the requirement
        is met to return a non-neutral result. 
         1  =   Long
         0  =   Neutral
        -1  =   Short

        Args:
            probabilities: List[float]
                The predicted probabilities for up and down. IMPORTANT: the provided list
                must contain a list with 2 items, the first one must represent the probability
                of the price going up and the second one the probabilities going down.

        Returns:
            Tuple[IPredictionResult, str] 
            (1|0|-1, 'long'|'neutral'|'short')
        """
        if probabilities[0] >= self.min_increase_probability:
            return 1, "long"
        elif probabilities[1] >= self.min_decrease_probability:
            return -1, "short"
        else:
            return 0, "neutral"







        






    def get_config(self) -> IProbabilityInterpreterConfig:
        """Returns the interpreter's data after having been initialized.

        Returns:
            IProbabilityInterpreterConfig
        """
        return { 
            "min_increase_probability": self.min_increase_probability,
            "min_decrease_probability": self.min_decrease_probability
        }