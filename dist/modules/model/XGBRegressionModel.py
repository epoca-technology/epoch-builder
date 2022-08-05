from typing import List, Union
from pandas import DataFrame
from modules.types import IModel, IPrediction, IPredictionMetaData
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.interpreter.PercentageChangeInterpreter import PercentageChangeInterpreter
from modules.prediction_cache.FeatureCache import FeatureCache
from modules.prediction_cache.PredictionCache import PredictionCache
from modules.model.ModelType import validate_id
from modules.model.Interface import RegressionModelInterface
from modules.xgb_regression.XGBRegression import XGBRegression






class XGBRegressionModel(RegressionModelInterface):
    """XGBRegressionModel Class
    
    This class is responsible of handling interactions with XGBoost Regression Models.

    Instance Properties:
        enable_cache: bool
            The state of the cache. If False, the model won't interact with the db.
        id: str
            The identifier of the saved keras model.
        regression: XGBRegression
            The instance of the XGBoost Regression.
        interpreter: PercentageChangeInterpreter
            The Interpreter instance that will be used to interpret Regression Predictions.
        feature_cache: FeatureCache
            The instance of the feature cache.
        prediction_cache: PredictionCache
            The instance of the prediction cache.
    """




    ## Initialization ## 


    def __init__(self, config: IModel, enable_cache: bool = False):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
            enable_cache: bool
                If enabled, the model will store predictions and features in the db.
        """
        # Init the cache state
        self.enable_cache: bool = enable_cache
        
        # Make sure there is 1 Regression Model
        if len(config["xgb_regressions"]) != 1:
            raise ValueError(f"A XGBRegressionModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['xgb_regressions'])}")

        # Initialize the ID of the model
        validate_id("XGBRegressionModel", config["id"])
        self.id: str = config["id"]

        # Initialize the regression
        self.regression: XGBRegression = XGBRegression(config["xgb_regressions"][0]["regression_id"])

        # Initialize the Interpreter Instance
        self.interpreter: PercentageChangeInterpreter = PercentageChangeInterpreter({
            "min_increase_change": 0,
            "min_decrease_change": 0
        })

        # Initialize the features cache instance
        self.feature_cache: FeatureCache = FeatureCache(self.id)

        # Initialize the prediction cache instance
        self.prediction_cache: PredictionCache = PredictionCache(self.id)






    ## Features ##



    def feature(self, current_timestamp: int, lookback_df: Union[DataFrame, None]=None) -> float:
        """In order to optimize performance, if cache is enabled, it will check the db
        before generating a feature. If the feature is not found, it will
        perform it and store it afterwards. If cache is not enabled, it will just 
        generate a traditional feature without storing the results.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            lookback_df: Union[DataFrame, None]
                Classifications can pass the Lookback DataFrame and it will be sliced 
                accordingly to match the model's lookback.
        
        Returns:
            float
        """
        # Initialize the adjusted lookback_df if provided
        df: Union[DataFrame, None] = self._get_adjusted_lookback_df(lookback_df)
        
        # Check if the cache is enabled
        if self.enable_cache:
            # Retrieve the candlestick range
            first_ot: int = 0
            last_ct: int = 0
            if isinstance(df, DataFrame):
                first_ot = int(df.iloc[0]["ot"])
                last_ct = int(df.iloc[-1]["ct"])
            else:
                first_ot, last_ct = Candlestick.get_lookback_prediction_range(self.regression.lookback, current_timestamp)

            # Retrieve the feature
            feature: Union[float, None] = self.feature_cache.get(first_ot, last_ct)

            # Check if the feature exists
            if feature == None:
                # Generate the feature
                feature = self._call_feature(current_timestamp)

                # Store it in cache
                self.feature_cache.save(first_ot, last_ct, feature)

                # Finally, return it
                return feature

            # If the feature exists, return it
            else:
                return feature

        # Otherwise, handle a traditional feature
        else:
            return self._call_feature(current_timestamp)






    def _call_feature(self, current_timestamp: int) -> float:
        """Given the current time, it will calculate the regression feature.

        Args:
            current_timestamp: int
                The current time in milliseconds.

        Returns:
            float
        """
        # Retrieve the normalized lookback df
        norm_df: DataFrame = Candlestick.get_lookback_df(self.regression.lookback, current_timestamp, normalized=True)

        # Generate the predictions
        preds: List[float] = self.regression.predict(norm_df["c"])

        # Calculate the price change from the current price to the last prediction and return the
        # normalized feature.
        return self._normalize_feature(Utils.get_percentage_change(norm_df["c"].iloc[-1], preds[-1]))





    def _normalize_feature(self, change: float) -> float:
        """Given a price prediction change, it will normalize the value
        to a -1 | 1 range based on the regression's properties.

        Args:
            change: float
                The percentage change between the current price and 
                the last predicted price.

        Returns:
            float
        """
        pass#@TODO











    ## Predictions ##



    def predict(self, current_timestamp: int, lookback_df: Union[DataFrame, None]=None) -> IPrediction:
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
        
        Returns:
            IPrediction
        """
        # Initialize the adjusted lookback_df if provided
        df: Union[DataFrame, None] = self._get_adjusted_lookback_df(lookback_df)
        
        # Check if the cache is enabled
        if self.enable_cache:
            # Retrieve the candlestick range
            first_ot: int = 0
            last_ct: int = 0
            if isinstance(df, DataFrame):
                first_ot = int(df.iloc[0]["ot"])
                last_ct = int(df.iloc[-1]["ct"])
            else:
                first_ot, last_ct = Candlestick.get_lookback_prediction_range(self.regression.lookback, current_timestamp)

            # Check if the prediction has already been cached
            pred: Union[IPrediction, None] = self.prediction_cache.get(first_ot, last_ct)

            # Check if the prediction exists
            if pred == None:
                # Generate the prediction
                pred = self._call_predict(current_timestamp, minimized_metadata=True)

                # Store it in cache
                self.prediction_cache.save(first_ot, last_ct, pred)

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
        preds: List[float] = self.regression.predict(norm_df["c"])

        # Prepend the current price to the predictions
        preds = [ norm_df["c"].iloc[-1] ] + preds

        # Interpret the predictions
        result, description = self.interpreter.interpret(preds)

        # Build the metadata
        metadata: IPredictionMetaData = { "d": description }
        if not minimized_metadata:
            metadata["pl"] = preds
        
        # Finally, return the prediction results
        return { "r": result, "t": int(current_timestamp), "md": [ metadata ] }









    ## Lookback DataFrame Adjustment ##




    def _get_adjusted_lookback_df(self, lookback_df: Union[DataFrame, None]) -> Union[DataFrame, None]:
        """Classifications can pass the lookback_df to the predict function. If so, the df needs to
        be adjusted to the Model's Instance Lookback since the Classification uses the Max Lookback
        from all the regressions in order to build the lookback_df.

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
        """Dumps the model's data into a dictionary.

        Args:
            None

        Returns:
            IModel
        """
        return {
            "id": self.id,
            "xgb_regressions": [{
                "regression_id": self.regression.id,
                "interpreter": self.interpreter.get_config(),
                "regression": self.regression.get_config()
            }]
        }






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a XGBRegressionModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        return isinstance(model.get("xgb_regressions"), list) \
                and len(model["xgb_regressions"]) == 1 \
                    and model.get("keras_regressions") == None \
                        and model.get("keras_classifications") == None \
                            and model.get("xgb_classifications") == None \
                                and model.get("consensus") == None