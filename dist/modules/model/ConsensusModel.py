from typing import List, Union
from pandas import DataFrame
from modules.types import IModel, IPrediction, IPredictionMetaData, IPredictionResult
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.interpreter.ConsensusInterpreter import ConsensusInterpreter
from modules.model.ModelType import validate_id, IModelType
from modules.model.Interface import ModelInterface
from modules.model.ClassificationModelFactory import ClassificationModelFactory, ClassificationModel




class ConsensusModel(ModelInterface):
    """ConsensusModel Class
    
    This class combines the predictions of multiple Classifications and predicts based
    on the provided interpreter's configuration.

    Instance Properties:
        enable_cache: bool
            The state of the cache. If False, the model won't interact with the db.
        id: str
            The identifier of the consensus model.
        sub_models: List[ClassificationModel]
            The instances of the classification models that will generate predictions.
        max_lookback: int
            The highest lookback among the models within.
        interpreter: ConsensusInterpreter
            The Interpreter instance that will be used to interpret Classification Model Predictions.
    """








    ## Initialization ## 



    def __init__(self, config: IModel, enable_cache: bool = False):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
            enable_cache: bool
                Placeholder. The consensus model does not make use of the cache functionality. Moreover,
                the classifications within use cache by default, as well as their inner regressions.

        Raises:
            ValueError:
                If any of the models' config properties are invalid.
                If there are less than 2 sub_models
                If the min_consensus is lower than 51% of the total models
        """
        # Init the cache state
        self.enable_cache: bool = enable_cache

        # Make sure there is 1 Classification Model
        if not isinstance(config["consensus"], dict):
            raise ValueError(f"The provided consensus configuration is invalid.")

        # Initialize the ID of the model
        validate_id("ConsensusModel", config["id"])
        self.id: str = config["id"]

        # Initialize the sub_models
        self.sub_models: List[ClassificationModel] = self._get_sub_models(config)

        # Initialize the max lookback
        self.max_lookback: int = max([m.get_lookback() for m in self.sub_models])

        # Initialize the Interpreter Instance
        self.interpreter: ConsensusInterpreter = ConsensusInterpreter(config["consensus"]["interpreter"])






    def _get_sub_models(self, config: IModel) -> List[ClassificationModel]:
        """Returns the list of model instances that will be used to generate predictions.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance

        Returns:
            List[ClassificationModel]

        Raises:
            ValueError:
                If any of the models' config properties are invalid.
                If there are less than 2 sub_models
                If the minimum consensus represents less than 51% of the models total
        """
        # Init the list
        models: List[ClassificationModel] = []

        # Check if there are any KerasClassificationModels
        if isinstance(config.get("keras_classifications"), list):
            if len(config["keras_classifications"]) > 0:
                for model_config in config["keras_classifications"]:
                    models.append(ClassificationModelFactory({
                        "id": model_config["classification_id"],
                        "keras_classifications": [{"classification_id": model_config["classification_id"]}]
                    }))
            else:
                raise ValueError("Received an empty list in the keras_classifications configuration.")

        # Check if there are any XGBClassificationModels
        if isinstance(config.get("xgb_classifications"), list):
            if len(config["xgb_classifications"]) > 0:
                for model_config in config["xgb_classifications"]:
                    models.append(ClassificationModelFactory({
                        "id": model_config["classification_id"],
                        "xgb_classifications": [{"classification_id": model_config["classification_id"]}]
                    }))
            else:
                raise ValueError("Received an empty list in the xgb_classifications configuration.")

        # Calculate the total models
        models_total: int = len(models)

        # Make sure at least 2 models were provided
        if models_total < 2:
            raise ValueError("A minimum of 2 models must be provided to a ConsensusModel.")

        # Make sure the minimum consensus is correct
        if Utils.get_percentage_out_of_total(config["consensus_model"]["interpreter"]["min_consensus"], models_total) < 51:
            raise ValueError("The minimum consensus must represent at least 51 percent of the total models.")

        # Finally, return the model instances
        return models








    ## Predictions ##




    def predict(self, current_timestamp: int, lookback_df: Union[DataFrame, None] = None) -> IPrediction:
        """Given the current time, it will perform a prediction and return it as 
        well as its metadata.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            lookback_df: Union[DataFrame, None]
                PLACEHOLDER: Not used by the ConsensusModel.
        
        Returns:
            IPrediction
        """
        # Initialize the list of prediction results and the metadata items
        results: List[IPredictionResult] = []
        metadata_items: List[IPredictionMetaData] = []

        # Initialize the lookback df
        lookback: DataFrame = Candlestick.get_lookback_df(self.max_lookback, current_timestamp)

        # Iterate over each sub model
        for model in self.sub_models:
            # Generate the prediction
            pred: IPrediction = model.predict(current_timestamp, lookback_df=lookback)

            # Append the result and the metadata
            results.append(pred["r"])
            metadata_items.append(pred["md"][0])

        # Interpret the predictions
        result, _ = self.interpreter.interpret(results)
        
        # Finally, return the prediction results
        return { "r": result, "t": int(current_timestamp), "md": metadata_items }















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
        # Initialize the partial model
        model: IModel = {
            "id": self.id,
            "consensus": {
                "sub_models": [],
                "interpreter": self.interpreter.get_config()
            }
        }

        # Iterate over each of the models and add the configurations accordingly
        for sub_model in self.sub_models:
            # Build the model's config
            config: IModel = sub_model.get_model()
            
            # Append it to the consensus sub models configuration
            model["consensus"]["sub_models"].append(config)

            # Initialize the type of model
            model_type: IModelType = type(sub_model).__name__

            # Append it to the Keras Classifications Config if applies
            if model_type == "KerasClassificationModel":
                if isinstance(model.get("keras_classifications"), list):
                    model["keras_classifications"].append(config["keras_classifications"][0])
                else:
                    model["keras_classifications"] = config["keras_classifications"]

            # Append it to the XGB Classifications Config if applies
            elif model_type == "XGBClassificationModel":
                if isinstance(model.get("xgb_classifications"), list):
                    model["xgb_classifications"].append(config["xgb_classifications"][0])
                else:
                    model["xgb_classifications"] = config["xgb_classifications"]

        # Finally, return the model's configuration
        return model






    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a ConsensusModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        return isinstance(model.get("consensus"), dict) and (
            isinstance(model.get("keras_classifications"), list) or
            isinstance(model.get("xgb_classifications"), list)
        )