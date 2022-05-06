from typing import TypedDict, List, Union, Dict, Any
from modules.arima import IArimaConfig





## Predictions ##



# Prediction Meta Data
# This is the data that was used by the interpreter to come up with a result.
# The only parameter that is required is the description (d) which should always
# follow the pattern 'long-*' or 'short-*'
class IPredictionMetaData(TypedDict):
    # Interpretation Description
    d: str

    # List of predictions generated by Arima. Only present in ArimaModel Predictions
    # when cache is disabled.
    pl: Union[List[float], None]

    # Arima Model Prediction Results. Only present in DecisionModels
    pr: Union[List[int], None]

    # Up Probability. Only present in DecisionModels
    up: Union[float, None]

    # Down Probability. Only present in DecisionModels
    dp: Union[float, None]




# Prediction
# The final prediction dict generated by the model. It contains the result, the time
# in which the prediction was made and the metadata.
# For ArimaModels and DecisionModels, the md list will always contain one element.
# On the other side, MultiDecisionModels contain as many metadata elements as decision models
# and they also have identical indexing.
class IPrediction(TypedDict):
    # Prediction result: -1 | 0 | 1
    r: int

    # The time in which the prediction was performed (milliseconds)
    t: int

    # Prediction metadata: A SingleModel will always output a single IPredictionMetaData
    # whereas, MultiModels will output any number of IPredictionMetaData dictionaries
    md: List[IPredictionMetaData]










## ArimaModel ##



# Arima Interpreter Configuration
# The configuration used in order to interpret the model's predictions based on the 
# percentual change from the first to the last prediction.
class IArimaModelInterpreterConfig(TypedDict):
    long: float
    short: float





# ArimaModel Configuration
# The configuration that will be use to generate and interpret predictions.
class IArimaModelConfig(TypedDict):
    # The number of prediction candlesticks that will look into the past in order to make a prediction.
    lookback: int

    # The number of predictions to be generated by Arima
    predictions: int

    # Parameters for ARIMA(p,d,q)(P,D,Q)m
    arima: IArimaConfig

    # The interpreter that will determine the prediction's result
    interpreter: IArimaModelInterpreterConfig










## DecisionModel ##




## MultiDecisionModel ##





## Model ##


# Model
# The final state of an ArimaModel, DecisionModel or MultiDecisionModel once an instance is initialized.
# ArimaModel: Only takes an id and a single element in the arima_models list. 
class IModel(TypedDict):
    # Identity of the Model. If it is an Arima Model, it must follow the guidelines.
    id: str

    # The number of decision models that must agree in order to output a non-neutral prediction result.
    # This property only exists in MultiDeceisionModel
    consensus: Union[int, None]

    # The minimum probability required to output a non-neutral prediction result. Only present in 
    # DecisionModels & MultiDecisionModels
    minimum_probability: Union[float, None]

    # The list of ArimaModels that will be used to predict accordingly based on the type of model.
    arima_models: List[IArimaModelConfig]

    # @TODO
    decision_models: Union[List[Any], None]







