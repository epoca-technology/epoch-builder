from typing import TypedDict, Tuple, Union
from modules._types.prediction_types import IPredictionResult




# Percent Change Interpreter Configuration
# The configuration used in order to interpret the model's predictions based on the 
# percentual change from the first to the last item.
class IPercentChangeInterpreterConfig(TypedDict):
    min_increase_change: float
    min_decrease_change: float








# Probability Interpreter Configuration
# The configuration used in order to interpret the model's predictions based on the 
# up and down probability.
class IProbabilityInterpreterConfig(TypedDict):
    min_increase_probability: float
    min_decrease_probability: float









# Consensus Interpreter Configuration
# The configuration used in order to interpret the model's predictions based on the 
# predictions generated by other models.
class IConsensusInterpreterConfig(TypedDict):
    min_consensus: float







# Interpreter Interface
# PercentageChangeInterpreter, ProbabilityInterpreter and ConsensusInterpreter implement the following interface
# in order to ensure compatibility across any of the processes.
class InterpreterInterface:
    # Interprets a prediction that has been generated by a model. Even though the function 
    # arguments can vary depending on the type of interpreter, the output will always be the 
    # same. 
    # prediction_result (1|0|-1) , result_description ('long'|'neutral'|'short')
    def interpret(self, *args) -> Tuple[IPredictionResult, str]:
        raise NotImplementedError("Interpreter.interpret has not been implemented.")


    # Retrieves the configuration of the Interpreter after being initialized
    def get_config(self) -> Union[IPercentChangeInterpreterConfig, IProbabilityInterpreterConfig, IConsensusInterpreterConfig]:
        raise NotImplementedError("Interpreter.get_config has not been implemented.")