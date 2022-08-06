from typing import List, Union
from pandas import DataFrame
from modules._types import IModel, IPrediction, IPredictionMetaData, ITechnicalAnalysis, ModelInterface
from modules.candlestick.Candlestick import Candlestick
from modules.interpreter.ProbabilityInterpreter import ProbabilityInterpreter
from modules.prediction_cache.PredictionCache import PredictionCache
from modules.model.ModelType import validate_id
from modules.model.RegressionModelFactory import RegressionModelFactory, RegressionModel
from modules.technical_analysis.TechnicalAnalysis import TechnicalAnalysis
from modules.keras_classification.KerasClassification import KerasClassification



class KerasClassificationModel(ModelInterface):
    """KerasClassificationModel Class
    
    This class is responsible of handling interactions with Keras Classification Models.

    Instance Properties:
        enable_cache: bool
            The state of the cache. If False, the model won't interact with the db.
        id: str
            The identifier of the saved keras model.
        classification: KerasClassification
            The instance of the Keras Classification Model.
        regressions: List[RegressionModel]
            The instances of the regression models that will be used to generate features.
        max_lookback: int
            The highest lookback among the regressions within.
        interpreter: ProbabilityInterpreter
            The Interpreter instance that will be used to interpret Classification Predictions.
        cache: PredictionCache
            The instance of the prediction cache.
    """





    ## Initialization ## 


    def __init__(self, config: IModel, enable_cache: bool = False):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
            enable_cache: bool
                If enabled, the model will store predictions in the db. Keep in mind that the regressions
                within the classification use cache by default.
        """
        # Init the cache state
        self.enable_cache: bool = enable_cache

        # Make sure there is 1 Classification Model
        if len(config["keras_classifications"]) != 1:
            raise ValueError(f"A KerasClassificationModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['keras_classifications'])}")

        # Initialize the ID of the model
        validate_id("KerasClassificationModel", config["id"])
        self.id: str = config["id"]

        # Initialize the classification
        self.classification: KerasClassification = KerasClassification(config["keras_classifications"][0]["classification_id"])

        # Initialize the Regression Instances
        self.regressions: List[RegressionModel] = [ RegressionModelFactory(m, True) for m in self.classification.regressions ]

        # Initialize the max lookback
        self.max_lookback: int = max([m.get_lookback() for m in self.regressions])

        # Initialize the Interpreter Instance
        self.interpreter: ProbabilityInterpreter = ProbabilityInterpreter({
            "min_increase_probability": 0,
            "min_decrease_probability": 0
        })

        # Initialize the prediction cache instance
        self.cache: PredictionCache = PredictionCache(self.id)







    ## Predictions ##




    def predict(self, current_timestamp: int, lookback_df: Union[DataFrame, None]=None) -> IPrediction:
        """In order to optimize performance, if cache is enabled, it will check the db
        or the temp cache before performing an actual prediction. If the prediction is not found, 
        it will perform it and store it afterwards. If cache is not enabled, it will just 
        perform a traditional prediction without storing the results.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            lookback_df: Union[DataFrame, None]
                The ConsensusModel can pass the lookback df to a Classification so then it
                can be spread across the regressions.
        
        Returns:
            IPrediction
        """
        # Check if the cache is enabled
        if self.enable_cache:
            # Retrieve the candlestick prediction range
            first_ot, last_ct = Candlestick.get_lookback_prediction_range(self.max_lookback, current_timestamp)

            # Check if the prediction has already been cached
            pred: Union[IPrediction, None] = self.cache.get(first_ot, last_ct)

            # Check if the prediction exists
            if pred == None:
                # Generate the prediction
                pred = self._call_predict(current_timestamp, lookback_df, minimized_metadata=True)

                # Store it in cache
                self.cache.save(first_ot, last_ct, pred)

                # Finally, return it
                return pred

            # If the prediction exists, return it
            else:
                return pred
        else:
            return self._call_predict(current_timestamp, lookback_df, minimized_metadata=False)
            







    def _call_predict(
        self, 
        current_timestamp: int, 
        lookback_df: Union[DataFrame, None], 
        minimized_metadata: bool
    ) -> IPrediction:
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
        # Build the features
        features: List[float] = self._get_features(current_timestamp, lookback_df)

        # Generate a prediction based on the features
        pred: List[float] = self.classification.predict(features)

        # Interpret the prediction
        result, description = self.interpreter.interpret(pred)

        # Build the metadata
        metadata: IPredictionMetaData = { "d": description }
        if not minimized_metadata:
            metadata["f"] = features
            metadata["up"] = pred[0]
            metadata["dp"] = pred[1]
        
        # Finally, return the prediction results
        return { "r": result, "t": int(current_timestamp), "md": [ metadata ] }







    def _get_features(self, current_timestamp: int, lookback_df: Union[DataFrame, None]) -> List[float]:
        """Builds the list of features that will be used by the Classification to predict.
        As well as dealing with Regression Predictions it will also build the TA values
        if enabled.

        Args:
            current_timestamp: int
                The open time of the current 1 minute candlestick.
            lookback_df: Union[DataFrame, None]
                ConsensusModels pass the lookback df for optimization reasons.

        Returns:
            List[float]
        """
        # Init the lookback_df
        lookback: DataFrame = Candlestick.get_lookback_df(self.max_lookback, current_timestamp) \
            if lookback_df is None else lookback_df

        # Generate predictions with all the regression models within the classification
        features: List[float] = [
            r.predict(current_timestamp, lookback_df=lookback)["r"] for r in self.regressions
        ]

        # Check if any Technical Anlysis feature needs to be added
        if self.classification.include_rsi or self.classification.include_aroon:
            # Retrieve the technical analysis
            ta: ITechnicalAnalysis = TechnicalAnalysis.get_technical_analysis(
                lookback,
                include_rsi=self.classification.include_rsi,
                include_aroon=self.classification.include_aroon
            )

            # Populate the RSI feature if enabled
            if self.classification.include_rsi:
                features.append(ta["rsi"])

            # Populate the Aroon feature if enabled
            if self.classification.include_aroon:
                features.append(ta["aroon"])

        # Finally, return all the features
        return features

















    ## General Retrievers ##






    def get_lookback(self) -> int:
        """Returns the lookback value of the model.

        Args:
            None

        Returns:
            int
        """
        return self.max_lookback

    







    def get_model(self) -> IModel:
        """Dumps the model's data into a dictionary.

        Args:
            None

        Returns:
            IModel
        """
        return {
            "id": self.id,
            "keras_classifications": [{
                "classification_id": self.classification.id,
                "interpreter": self.interpreter.get_config(),
                "classification": self.classification.get_config()
            }]
        }






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a KerasClassificationModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        return isinstance(model.get("keras_classifications"), list) \
                and len(model["keras_classifications"]) == 1\
                    and model.get("keras_regressions") == None \
                        and model.get("xgb_regressions") == None \
                            and model.get("xgb_classifications") == None \
                                and model.get("consensus") == None