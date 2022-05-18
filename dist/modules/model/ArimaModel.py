from typing import List, Union
from pandas import Series
from modules.candlestick import Candlestick
from modules.arima import Arima
from modules.interpreter import PercentageChangeInterpreter, IPercentChangeInterpreterConfig
from modules.prediction_cache import save_arima_pred, get_arima_pred
from modules.model import ModelInterface, IModel, IArimaModelConfig, IPrediction, IPredictionMetaData



class ArimaModel(ModelInterface):
    """ArimaModel Class

    Initializes an ArimaModel instance that is ready to perform predictions.

    Class Properties:
        DEFAULT_LOOKBACK: int
            The default value that will be used if the lookback isn't provided.
        DEFAULT_PREDICTIONS: int
            The default value that will be used if the predictions isn't provided.
        DEFAULT_INTERPRETER: IPercentChangeInterpreterConfig
            The default config that will be used if the interpreter isn't provided.

    Instance Properties:
        id: str
            The Arima identifier.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated by Arima.
        interpreter: PercentageChangeInterpreter
            The Interpreter instance that will be used to process Arima Predictions.
        arima: Arima
            The Arima Wrapper instance
    """
    # Default Lookback
    DEFAULT_LOOKBACK: int = 300

    # Default # of Predictions
    DEFAULT_PREDICTIONS: int = 10

    # Default Interpreter
    DEFAULT_INTERPRETER: IPercentChangeInterpreterConfig = { 'long': 0.05, 'short': 0.05 }





    ## Initialization ## 


    def __init__(self, config: IModel):
        """Initializes the instance properties as well as the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
        """
        # Make sure there is 1 Arima Model
        if len(config['arima_models']) != 1:
            raise ValueError(f"An ArimaModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['arima_models'])}")

        # Initialize the ID
        self.id: str = config['id']

        # Initialize the Model's Config
        model_config: IArimaModelConfig = config['arima_models'][0]

        # Initialize the lookback
        self.lookback: int = model_config['lookback'] \
            if isinstance(model_config.get('lookback'), int) else ArimaModel.DEFAULT_LOOKBACK

        # Initialize the number of predictions
        self.predictions: int = model_config['predictions'] \
            if isinstance(model_config.get('predictions'), int) else ArimaModel.DEFAULT_PREDICTIONS

        # Initialize the Interpreter Instance
        self.interpreter: PercentageChangeInterpreter = PercentageChangeInterpreter(model_config['interpreter'] \
            if isinstance(model_config.get('interpreter'), dict) else ArimaModel.DEFAULT_INTERPRETER)

        # Initialize the Arima Wrapper
        self.arima: Arima = Arima(model_config['arima'], self.predictions)

        # Validate the integrity of the model
        self._validate_integrity()













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
        # Check if the cache is enabled
        if enable_cache:
            # Retrieve the candlestick range
            first_ot, last_ct = Candlestick.get_lookback_prediction_range(self.lookback, current_timestamp)

            # Retrieve it from the database
            pred: Union[IPrediction, None] = get_arima_pred(
                self.id, 
                first_ot, 
                last_ct, 
                self.predictions, 
                self.interpreter.long,
                self.interpreter.short
            )

            # Check if the prediction does not exist
            if pred == None:
                # Generate it
                pred = self._call_predict(current_timestamp, minimized_metadata=True)

                # Store it in the database
                save_arima_pred(
                    self.id, 
                    first_ot, 
                    last_ct, 
                    self.predictions, 
                    self.interpreter.long,
                    self.interpreter.short,
                    pred
                )

                # Finally, return it
                return pred

            # Otherwise, return it
            else:
                return pred

        # Otherwise, handle a traditional prediction
        else:
            return self._call_predict(current_timestamp, minimized_metadata=False)






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
        # Retrieve the close prices series
        close_prices: Series = Candlestick.get_lookback_close_prices(self.lookback, current_timestamp)

        # Initialize Arima safely
        try:
            # Generate the predictions
            preds: List[float] = self.arima.predict(close_prices)

            # Interpret the predictions
            result, description = self.interpreter.interpret(preds)

            # Build the metadata
            metadata: IPredictionMetaData = { 'd': description }
            if not minimized_metadata:
                metadata['pl'] = preds
            
            # Finally, return the prediction results
            return { "r": result, "t": int(current_timestamp), "md": [ metadata ] }
        except Exception as e:
            print(f"{self.id} Prediction Error: {str(e)}")
            return { "r": 0, "t": int(current_timestamp), "md": [{'d': 'neutral-due-to-error: ' + str(e)}] }







    ## General Retrievers ##






    def get_lookback(self) -> int:
        """Returns the lookback value of the model.

        Args:
            None

        Returns:
            int
        """
        return self.lookback








    def get_model(self) -> IModel:
        """Dumps the model's data into a dictionary that will be used
        to get the insights based on its performance.

        Args:
            None

        Returns:
            IModel
        """
        return {
            "id": self.id,
            "arima_models": [{
                "arima": self.arima.config,
                "lookback": self.lookback,
                "predictions": self.predictions,
                "interpreter": self.interpreter.get_config(),
            }]
        }






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is an ArimaModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        return ArimaModel._is_id(model['id']) \
            and isinstance(model.get('arima_models'), list) \
                and len(model['arima_models']) == 1 \
                    and model.get('regression_models') == None \
                        and model.get('classification_models') == None









    ## Model Integrity Validation ##
    


    def _validate_integrity(self) -> None:
        """Verifies if the ID follows the ApdqPDQm guidelines. It will also make sure that its
        configuration matches the arima configuration from the ID.

        Raises:
            ValueError:
                If the ID of the model does not follow the Apdq or ApdqPDQm guideline.
                If there is a missmatch between the config in the ID and the actual
                    Arima configuration.
        """
        # Make sure the ID follows the proper guidelines
        if not ArimaModel._is_id(self.id):
            raise ValueError(f"The ID of the arima model must follow the Apdq or ApdqPDQm guideline. Received {self.id}")

        # Split the ID into chunks and save the Arima Configuration chunk
        chunk: str = self.id.split('A')[1]

        # Make sure the p, d and q values match perfectly
        if int(chunk[0]) != self.arima.config['p']:
            raise ValueError(f"Arima Configuration Missmatch for p. Received {chunk[0]} and {self.arima.config['p']}")
        if int(chunk[1]) != self.arima.config['d']:
            raise ValueError(f"Arima Configuration Missmatch for d. Received {chunk[1]} and {self.arima.config['d']}")
        if int(chunk[2]) != self.arima.config['q']:
            raise ValueError(f"Arima Configuration Missmatch for q. Received {chunk[2]} and {self.arima.config['q']}")

        # Check if it is a Sarima Model
        if len(chunk) == 7:
            # Make sure the P, D, Q and m values match perfectly
            if int(chunk[3]) != self.arima.config['P']:
                raise ValueError(f"Sarima Configuration Missmatch for P. Received {chunk[3]} and {self.arima.config['P']}")
            if int(chunk[4]) != self.arima.config['D']:
                raise ValueError(f"Sarima Configuration Missmatch for D. Received {chunk[4]} and {self.arima.config['D']}")
            if int(chunk[5]) != self.arima.config['Q']:
                raise ValueError(f"Sarima Configuration Missmatch for Q. Received {chunk[5]} and {self.arima.config['Q']}")
            if int(chunk[6]) != self.arima.config['m']:
                raise ValueError(f"Sarima Configuration Missmatch for m. Received {chunk[6]} and {self.arima.config['m']}")





    @staticmethod
    def _is_id(id: str) -> bool:
        """Checks if a string is a valid Arima Model ID.

        Args:
            id: str
                The ID of the model.

        Returns:
            bool
        """
        return isinstance(id, str) and id[0] == 'A' and (len(id) == 4 or len(id) == 8)