from typing import List, Tuple
from modules.types import IProbabilityInterpreterConfig
from modules.interpreter.Interface import InterpreterInterface




class ProbabilityInterpreter(InterpreterInterface):
    """ProbabilityInterpreter Class

    This class takes the up and down probabilities and performs an interpretation.

    Class Properties:
        MIN_PROBABILITY_VALUE: float
            The minimum value allowed for the min probability.

    Instance Properties:
        min_probability: float
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
                If the min_probability config is invalid.
        """
        # Validate the min_probability configuration
        if not isinstance(config['min_probability'], float) \
            or config['min_probability'] < ProbabilityInterpreter.MIN_PROBABILITY_VALUE \
                or config['min_probability'] >= ProbabilityInterpreter.MAX_PROBABILITY_VALUE:
            raise ValueError(f"The min_probability config must be a valid float greater than \
                {ProbabilityInterpreter.MIN_PROBABILITY_VALUE} and smaller than {ProbabilityInterpreter.MAX_PROBABILITY_VALUE}. \
                Received: {str(config['min_probability'])}")

        # Initialize the instance properties
        self.min_probability: float = config["min_probability"]









 



    def interpret(self, probabilities: List[float]) -> Tuple[int, str]:
        """Given a list of probabilities for up and down, it will check if the requirement
        is met to return a non-neutral result. 
         1  =   Long
         0  =   Neutral
        -1  =   Short

        Args:
            probabilities: List[float]
                The predicted probabilities for up and down.

        Returns:
            Tuple[int, str] (1|0|-1, 'long'|'neutral'|'short')
        
        Raises:
            ValueError: 
                If the length of the probabilities is different to 2
        """
        # Make sure there are at least 5 items in the predictions list
        if len(probabilities) != 2:
            raise ValueError(f"A probability interpretation requires the probability for up and down. Received {str(probabilities)}")
        
        # Return the packed results accordingly
        if probabilities[0] >= self.min_probability:
            return 1, "long"
        elif probabilities[1] >= self.min_probability:
            return -1, "short"
        else:
            return 0, "neutral"







        






    def get_config(self) -> IProbabilityInterpreterConfig:
        """Returns the interpreter's data after having been initialized.

        Returns:
            IProbabilityInterpreterConfig
        """
        return { "min_probability": self.min_probability }