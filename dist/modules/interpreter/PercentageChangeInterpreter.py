from typing import List, Tuple
from modules.utils import Utils
from modules.interpreter import IPercentChangeInterpreterConfig, InterpreterInterface




class PercentageChangeInterpreter(InterpreterInterface):
    """PercentageChangeInterpreter Class

    This class takes a series of predictions and performs an interpretation based on the
    percentage change found between the first and the last predictions.

    Class Properties:
        MIN_CHANGE_VALUE: float
            The minimum value the long and the short configs can have

    Instance Properties:
        long: float
            The percentage change that will be used in order to predict a long position. Changes under this
            value will be considered neutral.
        short: float
            The percentage change that will be used in order to predict a short position. Even though the 
            required value is a positive number, it will be converted to negative in order to interpret
            the position. Changes above this value will be considered neutral.
    """

    # Minimum percentage change value for long and short configuration
    MIN_CHANGE_VALUE: float = 0.05

  




    def __init__(self, config: IPercentChangeInterpreterConfig):
        """Initializes the Interpreter Class based on the provided configuration.

        Args:
            config: IPercentChangeInterpreterConfig
                The configuration to interpret predictions.

        Raises:
            ValueError:
                If the long config is invalid.
                If the short config is invalid.
        """
        # Validate the long configuration
        if not isinstance(config['long'], (int, float)) or config['long'] < PercentageChangeInterpreter.MIN_CHANGE_VALUE:
            raise ValueError(f"The long config must be a valid float greater than {PercentageChangeInterpreter.MIN_CHANGE_VALUE}. \
                Received: {str(config['long'])}")

        # Validate the short configuration
        if not isinstance(config['short'], (int, float)) or config['short'] < PercentageChangeInterpreter.MIN_CHANGE_VALUE:
            raise ValueError(f"The short config must be a valid float greater than {PercentageChangeInterpreter.MIN_CHANGE_VALUE}. \
                Received: {str(config['short'])}")

        # Initialize the instance properties
        self.long: float = config['long']
        self.short: float = config['short']









 



    def interpret(self, predictions: List[float]) -> Tuple[int, str]:
        """Given a list of predictions, it will calculate the change between the 
        first and the last. Based on this result and the config values provided will
        determine a result.
         1  =   Long
         0  =   Neutral
        -1  =   Short

        Args:
            predictions: List[float]
                The list of predictions generated by the Arima Model.

        Returns:
            Tuple[int, str] (1|0|-1, 'long'|'neutral'|'short')
        
        Raises:
            ValueError: 
                If the length of the predictions list is less than 5
        """
        # Make sure there are at least 5 items in the predictions list
        if len(predictions) < 5:
            raise ValueError(f"An interpretation requires a minimum of 5 predictions. Received {len(predictions)}")

        # Calculate the percentual change between the first and the last prediction
        change: float = Utils.get_percentage_change(predictions[0], predictions[-1])
        
        # Return the packed results accordingly
        if change >= self.long:
            return 1, 'long'
        elif change <= -(self.short):
            return -1, 'short'
        else:
            return 0, 'neutral'







        






    def get_config(self) -> IPercentChangeInterpreterConfig:
        """Returns the interpreter's data after having been initialized.

        Returns:
            IPercentChangeInterpreterConfig
        """
        return { 'long': self.long, 'short': self.short }