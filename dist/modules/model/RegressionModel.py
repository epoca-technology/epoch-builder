from typing import List, Union
from pandas import DataFrame
from modules.candlestick import Candlestick
from modules.regression import Regression
from modules.interpreter import PercentageChangeInterpreter
from modules.prediction_cache import TemporaryPredictionCache
from modules.model import ModelInterface, IModel, IPrediction, IPredictionMetaData, IRegressionModelConfig






class RegressionModel(ModelInterface):
    """RegressionModel Class
    
    This class is responsible of handling interactions with Keras Regression Models.

    Instance Properties:
        id: str
            The identifier of the saved keras model.
        regression: Regression
            The instance of the Keras Regression Model.
        interpreter: PercentageChangeInterpreter
            The Interpreter instance that will be used to interpret Regression Predictions.
        cache: TemporaryPredictionCache
            The instance of the prediction temporary cache.

    """


    ## Initialization ## 


    def __init__(self, config: IModel):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
        """
        # Make sure there is 1 Regression Model
        if len(config['regression_models']) != 1:
            raise ValueError(f"A RegressionModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['regression_models'])}")

        # Initialize the ID of the model
        self.id: str = config['id']

        # Initialize the Model's Config
        model_config: IRegressionModelConfig = config['regression_models'][0]

        # Initialize the regression
        self.regression: Regression = Regression(model_config['regression_id'])

        # Initialize the Interpreter Instance
        self.interpreter: PercentageChangeInterpreter = PercentageChangeInterpreter(model_config['interpreter'])

        # Initialize the prediction cache instance
        self.cache: TemporaryPredictionCache = TemporaryPredictionCache()





    ## Predictions ##



    def predict(
        self, 
        current_timestamp: int, 
        lookback_df: Union[DataFrame, None] = None, 
        enable_cache: bool = False
    ) -> IPrediction:
        """In order to optimize performance, if cache is enabled, it will check the db
        before performing an actual prediction. If the prediction is not found, it will
        perform it and store it afterwards. If cache is not enabled, it will just 
        perform a traditional prediction without storing the results.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            lookback_df: Union[DataFrame, None]
                Classifications can pass the Lookback DataFrame and it will be sliced 
                accordingly to match the model's lookback.
            enable_cache: bool
                If true, it will check the db before calling the actual predict method.
        
        Returns:
            IPrediction
        """
        # Initialize the adjusted lookback_df if provided
        df: Union[DataFrame, None] = self._get_adjusted_lookback_df(lookback_df)
        
        # Check if the cache is enabled
        if enable_cache:
            # Retrieve the candlestick range
            first_ot: int = 0
            last_ct: int = 0
            if isinstance(df, DataFrame):
                first_ot = int(df.iloc[0]["ot"])
                last_ct = int(df.iloc[-1]["ct"])
            else:
                first_ot, last_ct = Candlestick.get_lookback_prediction_range(self.regression.lookback, current_timestamp)

            # Check if the prediction has already been cached
            pred: Union[IPrediction, None] = self.cache.get(first_ot, last_ct)

            # Check if the prediction exists
            if pred == None:
                # Generate the prediction
                pred = self._call_predict(current_timestamp, minimized_metadata=True)

                # Store it in cache
                self.cache.save(first_ot, last_ct, pred)

                # Finally, return it
                return pred

            # If the prediction exists, return it
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
        # Retrieve the normalized lookback df
        norm_df: DataFrame = Candlestick.get_lookback_df(self.regression.lookback, current_timestamp, normalized=True)

        # Generate the predictions
        preds: List[float] = self.regression.predict(norm_df)

        # Interpret the predictions
        result, description = self.interpreter.interpret(preds)

        # Build the metadata
        metadata: IPredictionMetaData = { 'd': description }
        if not minimized_metadata:
            metadata['npl'] = preds
        
        # Finally, return the prediction results
        return { "r": result, "t": int(current_timestamp), "md": [ metadata ] }






    def _get_adjusted_lookback_df(self, lookback_df: Union[DataFrame, None]) -> Union[DataFrame, None]:
        """Classifications can pass the lookback_df to the predict function. If so, the df needs to
        be adjusted to the Model's Instance Lookback since the Classification uses the Max Lookback
        from all the models in order to build the lookback_df.

        Args:
            lookback_df: Union[DataFrame, None]
                The DataFrame to be adjusted if provided.
        
        Returns:
            Union[DataFrame, None]
        """
        return lookback_df.iloc[-self.regression.lookback:] if isinstance(lookback_df, DataFrame) else None











    ## General Retrievers ##






    def get_lookback(self) -> int:
        """Returns the lookback value of the model.

        Args:
            None

        Returns:
            int
        """
        return self.regression.lookback

    







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
            "regression_models": [{
                "regression_id": self.regression.id,
                "interpreter": self.interpreter.get_config(),
                "regression": self.regression.get_config()
            }]
        }






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a RegressionModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        return isinstance(model.get('regression_models'), list) \
                and len(model['regression_models']) == 1 \
                    and model.get('arima_models') == None \
                        and model.get('classification_models') == None