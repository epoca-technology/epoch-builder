from typing import List, Union
from numpy import around
from pmdarima import ARIMA
from modules.candlestick import Candlestick
from modules.database import save_prediction, get_prediction
from modules.model import IModel, Interpreter, IPrediction, IPredictionMetaData, IArimaConfig



class SingleModel:
    """SingleModel Class

    Initializes a SingleModel instance that is ready to perform predictions.

    Instance Properties:
        id: str
            The name of the model that makes it identifiable.
        arima: IArimaConfig
            The configuration that will be used to generate predictions.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        interpreter: Interpreter
            The Interpreter instance that will be used to process Arima Predictions.
    """


    def __init__(self, config: IModel):
        """Initializes the instance properties as well as the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
        """
        # Make sure there is 1 single model
        if len(config['single_models']) != 1:
            raise ValueError(f"A SingleModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['single_models'])}")

        # Initialize the instance properties
        self.id = config['id']
        self.arima = self._get_arima_config(config['single_models'][0]['arima'])
        self.lookback = config['single_models'][0]['lookback']

        # Initialize the Interpreter Instance
        self.interpreter = Interpreter(config['single_models'][0]['interpreter'])





    def _get_arima_config(self, config: IArimaConfig) -> IArimaConfig:
        """Builds the Arima Config Dictionary. Optional values are filled with
        zeros if not provided.

        Args:
            config: IArimaConfig
                The configuration dict provided when initializing the instance.

        Returns:
            IArimaConfig
        """
        return {
            "predictions": config['predictions'],
            "p": config['p'],
            "d": config['d'],
            "q": config['q'],
            "P": config['P'] if isinstance(config.get('P'), int) else 0,
            "D": config['D'] if isinstance(config.get('D'), int) else 0,
            "Q": config['Q'] if isinstance(config.get('Q'), int) else 0,
            "m": config['m'] if isinstance(config.get('m'), int) else 0,
        }





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
            first_ot, last_ct = Candlestick.get_current_prediction_range(self.lookback, current_timestamp)

            # Retrieve it from the database
            pred: Union[IPrediction, None] = get_prediction(self.id, first_ot, last_ct)

            # Check if the prediction does not exist
            if pred == None:
                # Generate it
                pred = self._call_predict(current_timestamp)

                # Store it in the database
                save_prediction(self.id, first_ot, last_ct, pred)

                # Finally, return it
                return pred

            # Otherwise, return it
            else:
                return pred

        # Otherwise, handle a traditional prediction
        else:
            return self._call_predict(current_timestamp)






    def _call_predict(self, current_timestamp: int) -> IPrediction:
        """Given the current time, it will perform a prediction and return it as 
        well as its metadata.

        Args:
            current_timestamp: int
                The current time in milliseconds.

        Returns:
            IPrediction
        """
        # Retrieve the prediction data
        series, rsi, short_ema, long_ema = Candlestick.get_data_to_predict_on(
            current_timestamp=current_timestamp,
            lookback=self.lookback,
            include_rsi=self.interpreter.rsi['active'],
            include_ema=self.interpreter.ema['active'],
        )

        # Initialize Arima safely
        try:
            # Initialize the Arima Model
            arima_model: ARIMA = ARIMA(
                order=(self.arima['p'], self.arima['d'], self.arima['q']), 
                seasonal_order=(self.arima['P'], self.arima['D'], self.arima['Q'], self.arima['m']),
                suppress_warnings=True
            )

            # Fit the model to the retrieved series
            arima_model.fit(series)

            # Generate the predictions
            preds: List[float] = around(arima_model.predict(self.arima['predictions']), decimals=2).tolist()

            # Interpret the predictions
            result, description = self.interpreter.get_interpretation(preds, rsi, short_ema, long_ema)
            
            # Finally, return the prediction results
            return {
                "r": result,
                "t": int(current_timestamp),
                "md": [self._get_prediction_metadata(preds, description, rsi, short_ema, long_ema)]
            }
        except Exception as e:
            print(f"Arima Prediction Error: {str(e)}")
            return {
                "r": 0,
                "t": int(current_timestamp),
                "md": [{'d': 'neutral-due-to-error: ' + str(e)}]
            }






    def _get_prediction_metadata(
        self,
        prediction_list: List[float],
        interpretation_description: str,
        rsi: Union[float, None],
        short_ema: Union[float, None],
        long_ema: Union[float, None]
    ) -> IPredictionMetaData:
        """Given all the prediction metadata, it will put it together in a dictionary.

        Args:
            prediction_list: List[float]
                The list of predictions generated by Arima.
            interpretation_description: str
                The description provided by the interpreter.
            rsi: Union[float, None]
                The RSI Value if the indicator is enabled.
            short_ema: Union[float, None]
                The Short EMA Value if the indicator is enabled.
            long_ema: Union[float, None]
                The Long EMA Value if the indicator is enabled.

        Returns:
            IPredictionMetaData
        """
        # Init the metadata dict
        md: IPredictionMetaData = { "pl": prediction_list, "d": interpretation_description }

        # Check if the RSI should be added
        if self.interpreter.rsi['active']:
            md['rsi'] = rsi

        # Check if the EMA should be added
        if self.interpreter.ema['active']:
            md['sema'] = short_ema
            md['lema'] = long_ema

        # Finally, return the metadata
        return md







    def get_max_lookback(self) -> int:
        """Returns the lookback value of the model. In the case of a MultiModel
        instance, it needs to override this method and find the highest lookback
        value from the Model list.

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
            "single_models": [{
                "arima": self.arima,
                "lookback": self.lookback,
                "interpreter": self.interpreter.get_interpreter(),
            }]
        }