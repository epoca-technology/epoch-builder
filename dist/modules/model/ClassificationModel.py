from typing import List, Union
from pandas import DataFrame
from modules._types import IModel, IPrediction, IPredictionMetaData, ModelInterface
from modules.candlestick.Candlestick import Candlestick
from modules.interpreter.ProbabilityInterpreter import ProbabilityInterpreter
from modules.prediction_cache.PredictionCache import PredictionCache
from modules.model.RegressionModelFactory import RegressionModelFactory, RegressionModelInstance
from modules.keras_classification.KerasClassification import KerasClassification
from modules.xgb_classification.XGBClassification import XGBClassification
from modules.model.ClassificationFeatures import build_features





# Class
class ClassificationModel(ModelInterface):
    """ClassificationModel Class
    
    This class implements a ClassificationModel that must be extended by the supported 
    technologies that offer a Classification Solution.

    Instance Properties:
        enable_cache: bool
            The state of the cache. If False, the model won't interact with the db.
        id: str
            The identifier of the saved keras model.
        classification: Union[KerasClassification, XGBClassification]
            The instance of the Keras Classification Model.
        regressions: List[RegressionModelInstance]
            The instances of the regression models that will be used to generate features.
        max_lookback: int
            The highest lookback among the regressions within.
        interpreter: ProbabilityInterpreter
            The Interpreter instance that will be used to interpret Classification Predictions.
        cache: PredictionCache
            The instance of the prediction cache.
    """





    ## Initialization ## 


    def __init__(self, config: IModel, classification: Union[KerasClassification, XGBClassification], enable_cache: bool = False):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
            classification: Union[KerasClassification, XGBClassification]
                The classification that will be used to generate predictions.
            enable_cache: bool
                If enabled, the model will store predictions in the db. Keep in mind that the regressions
                within the classification use cache by default.
        """
        # Init the cache state
        self.enable_cache: bool = enable_cache

        # Initialize the ID of the model
        self.id: str = config["id"]

        # Initialize the classification
        self.classification: Union[KerasClassification, XGBClassification] = classification

        # Initialize the Regression Instances
        self.regressions: List[RegressionModelInstance] = [ RegressionModelFactory(m, True) for m in self.classification.regressions ]

        # Initialize the max lookback
        self.max_lookback: int = max([m.get_lookback() for m in self.regressions])

        # Initialize the Interpreter Instance
        self.interpreter: ProbabilityInterpreter = ProbabilityInterpreter({
            "min_increase_probability": self.classification.discovery["increase_successful_mean"],
            "min_decrease_probability": self.classification.discovery["decrease_successful_mean"]
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
        features: List[float] = build_features(
            current_timestamp=current_timestamp, 
            regressions=self.regressions, 
            max_lookback=self.max_lookback, 
            include_rsi=self.classification.include_rsi,
            include_aroon=self.classification.include_aroon,
            lookback_df=lookback_df
        )

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
        raise NotImplementedError("ClassificationModel.get_model has not been properly implemented.")






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a KerasClassificationModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        raise NotImplementedError("ClassificationModel.is_config has not been properly implemented.")