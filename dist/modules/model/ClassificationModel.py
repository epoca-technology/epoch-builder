from typing import List
from pandas import DataFrame
from modules.candlestick import Candlestick
from modules.regression import Regression
from modules.interpreter import ProbabilityInterpreter
from modules.model import ModelInterface, IModel, IPrediction, IPredictionMetaData, \
    IArimaModelConfig, IRegressionModelConfig



class ClassificationModel(ModelInterface):
    """ClassificationModel Class
    
    This class is responsible of handling interactions with Keras Classification Models.

    Class Properties:
        ...

    Instance Properties:
        id: str
            The identifier of the saved keras model.
        classification: Classification
            The instance of the Keras Classification Model.
        interpreter: ProbabilityInterpreter
            The Interpreter instance that will be used to interpret Classification Predictions.

    """





    ## Initialization ## 


    def __init__(self, config: IModel):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
        """
        pass








    ## Predictions ##




    def predict(self, current_timestamp: int, enable_cache: bool = False) -> IPrediction:
        """In order to optimize performance, if cache is enabled, it will check the db
        before performing an actual prediction. If the prediction is not found, it will
        perform it and store it afterwards. If cache is not enabled, it will just 
        perform a traditional prediction without storing the results.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            enable_cache: bool
                If true, it will check the db before calling the actual predict method.
        
        Returns:
            IPrediction
        """
        pass







    def _call_predict(self, current_timestamp: int, minimized_metadata: bool) -> IPrediction:
        """Given the current time, it will perform a prediction and return it as 
        well as its metadata.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            minimized_metadata: bool
                If this property is enabled, the metadata will only include the description.

        Returns:
            IPrediction
        """
        pass








    ## General Retrievers ##






    def get_lookback(self) -> int:
        """Returns the lookback value of the model.

        Args:
            None

        Returns:
            int
        """
        pass

    







    def get_model(self) -> IModel:
        """Dumps the model's data into a dictionary that will be used
        to get the insights based on its performance.

        Args:
            None

        Returns:
            IModel
        """
        pass






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a ClassificationModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        # Initialize the regression models lists. Handling potential None values
        arima: List[IArimaModelConfig] = model['arima_models'] \
            if model.get('arima_models') is not None else []
        regression: List[IArimaModelConfig] = model['regression_models'] \
            if model.get('regression_models') is not None else []

        # Check if the provided model configuration matches
        return isinstance(model.get('classification_models'), list) \
                and len(model['classification_models']) == 1 \
                    and (len(arima) + len(regression)) >= 5