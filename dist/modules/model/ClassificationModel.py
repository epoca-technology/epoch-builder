from typing import List, Union
from pandas import DataFrame
from modules.candlestick import Candlestick
from modules.interpreter import ProbabilityInterpreter
from modules.model import ModelInterface, IModel, RegressionModelFactory, IPrediction, \
    IPredictionMetaData, IClassificationModelConfig, IArimaModelConfig, ArimaModel, RegressionModel
from modules.technical_analysis import TechnicalAnalysis, ITechnicalAnalysis
from modules.classification.Classification import Classification



class ClassificationModel(ModelInterface):
    """ClassificationModel Class
    
    This class is responsible of handling interactions with Keras Classification Models.

    Instance Properties:
        id: str
            The identifier of the saved keras model.
        classification: Classification
            The instance of the Keras Classification Model.
        regressions: List[Union[ArimaModel, RegressionModel]]
            The instances of the regression models that will be used to generate features.
        max_lookback: int
            The highest lookback among the regressions within.
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
        # Make sure there is 1 Classification Model
        if len(config["classification_models"]) != 1:
            raise ValueError(f"A ClassificationModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['classification_models'])}")

        # Initialize the ID of the model
        self.id: str = config['id']

        # Initialize the Model's Config
        model_config: IClassificationModelConfig = config["classification_models"][0]

        # Initialize the classification
        self.classification: Classification = Classification(model_config['classification_id'])

        # Initialize the Regression Instances
        self.regressions: List[Union[ArimaModel, RegressionModel]] = [
            RegressionModelFactory(m) for m in self.classification.regressions
        ]

        # Initialize the max lookback
        self.max_lookback: int = max([m.get_lookback() for m in self.regressions])

        # Initialize the Interpreter Instance
        self.interpreter: ProbabilityInterpreter = ProbabilityInterpreter(model_config['interpreter'])







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
                Placeholder. This property is only used by ArimaModel|RegressionModel for
                optimization reasons. Classifications ignore this value.
            enable_cache: bool
                If true, it will check the db before calling the actual predict method.
        
        Returns:
            IPrediction
        """
        # Check if the cache is enabled
        if enable_cache:
            # @TODO
            return self._call_predict(current_timestamp, minimized_metadata=True)
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
        # Build the features
        features: List[float] = self._get_features(current_timestamp)

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







    def _get_features(self, current_timestamp: int) -> List[float]:
        """Builds the list of features that will be used by the Classification to predict.
        As well as dealing with Regression Predictions it will also build the TA values
        if enabled.

        Args:
            current_timestamp: int
                The open time of the current 1 minute candlestick.

        Returns:
            List[float]
        """
        # Init the lookback_df
        lookback_df: DataFrame = Candlestick.get_lookback_df(self.max_lookback, current_timestamp)

        # Generate predictions with all the regression models within the classification
        features: List[float] = [
            r.predict(
                current_timestamp, 
                lookback_df=lookback_df,
                enable_cache=True
            )["r"] for r in self.regressions
        ]

        # Check if any Technical Anlysis feature needs to be added
        if self.classification.include_rsi or self.classification.include_aroon:
            # Retrieve the technical analysis
            ta: ITechnicalAnalysis = TechnicalAnalysis.get_technical_analysis(
                lookback_df,
                include_rsi=self.classification.include_rsi,
                include_aroon=self.classification.include_aroon
            )

            # Populate the RSI feature if enabled
            if self.classification.include_rsi:
                features.append(ta["rsi"])

            # Populate the Aroon features if enabled
            if self.classification.include_aroon:
                features.append(ta["aroon_up"])
                features.append(ta["aroon_down"])

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
        """Dumps the model's data into a dictionary that will be used
        to get the insights based on its performance.

        Args:
            None

        Returns:
            IModel
        """
        return {
            "id": self.id,
            "classification_models": [{
                "classification_id": self.classification.id,
                "interpreter": self.interpreter.get_config(),
                "classification": self.classification.get_config()
            }]
        }






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