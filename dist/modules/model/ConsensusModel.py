from typing import List, Union
from pandas import DataFrame
from modules.types import IModel, IPrediction, IPredictionMetaData
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.interpreter.ConsensusInterpreter import ConsensusInterpreter
from modules.model.Interface import ModelInterface
from modules.model.ArimaModel import ArimaModel
from modules.model.RegressionModel import RegressionModel
from modules.model.ClassificationModel import ClassificationModel



class ConsensusModel(ModelInterface):
    """ConsensusModel Class
    
    This class is responsible of handling interactions with any number of models except for itself.

    Instance Properties:
        id: str
            The identifier of the consensus model.
        sub_models: List[Union[ArimaModel, RegressionModel, ClassificationModel]]
            The instances of the single models that will generate predictions.
        max_lookback: int
            The highest lookback among the models within.
        interpreter: ConsensusInterpreter
            The Interpreter instance that will be used to interpret Single Model Predictions.
    """








    ## Initialization ## 



    def __init__(self, config: IModel):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance

        Raises:
            ValueError:
                If any of the models' config properties are invalid.
                If there are less than 2 sub_models
                If the min_consensus is lower than 51% of the total models
        """
        # Make sure there is 1 Classification Model
        if not isinstance(config["consensus_model"], dict):
            raise ValueError(f"The provided consensus_model configuration is invalid.")

        # Initialize the ID of the model
        self.id: str = config["id"]

        # Initialize the sub_models
        self.sub_models: List[Union[ArimaModel, RegressionModel, ClassificationModel]] = self._get_sub_models(config)

        # Initialize the max lookback
        self.max_lookback: int = max([m.get_lookback() for m in self.sub_models])

        # Initialize the Interpreter Instance
        self.interpreter: ConsensusInterpreter = ConsensusInterpreter(config["consensus_model"]["interpreter"])






    def _get_sub_models(self, config: IModel) -> List[Union[ArimaModel, RegressionModel, ClassificationModel]]:
        """Returns the list of model instances that will be used to generate predictions.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance

        Raises:
            ValueError:
                If any of the models' config properties are invalid.
                If there are less than 2 sub_models
                If the minimum consensus represents less than 51% of the models total
        """
        # Init the list
        models: List[Union[ArimaModel, RegressionModel, ClassificationModel]] = []

        # Check if there are any ArimaModels
        if isinstance(config.get("arima_models"), list):
            if len(config["arima_models"]) > 0:
                for model_config in config["arima_models"]:
                    models.append(ArimaModel({
                        "id": f"A{model_config['arima']['p']}{model_config['arima']['d']}{model_config['arima']['q']}",
                        "arima_models": [{"arima": model_config["arima"], "interpreter": model_config["interpreter"]}]
                    }))
            else:
                raise ValueError("Received an empty list in the arima_models configuration.")

        # Check if there are any RegressionModels
        if isinstance(config.get("regression_models"), list):
            if len(config["regression_models"]) > 0:
                for model_config in config["regression_models"]:
                    models.append(RegressionModel({
                        "id": model_config["regression_id"],
                        "regression_models": [{"regression_id": model_config["regression_id"], "interpreter": model_config["interpreter"]}]
                    }))
            else:
                raise ValueError("Received an empty list in the regression_models configuration.")

        # Check if there are any ClassificationModels
        if isinstance(config.get("classification_models"), list):
            if len(config["classification_models"]) > 0:
                for model_config in config["classification_models"]:
                    models.append(ClassificationModel({
                        "id": model_config["classification_id"],
                        "classification_models": [{"classification_id": model_config["classification_id"], "interpreter": model_config["interpreter"]}]
                    }))
            else:
                raise ValueError("Received an empty list in the classification_models configuration.")

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




    def predict(
        self, 
        current_timestamp: int, 
        lookback_df: Union[DataFrame, None] = None, # Placeholder
        enable_cache: bool = False # Placeholder
    ) -> IPrediction:
        """Given the current time, it will perform a prediction and return it as 
        well as its metadata.

        Args:
            current_timestamp: int
                The current time in milliseconds.
            lookback_df: Union[DataFrame, None]
                PLACEHOLDER: Not used by the ConsensusModel.
            enable_cache: bool
                PLACEHOLDER: Not used by the ConsensusModel.
        
        Returns:
            IPrediction
        """
        # Initialize the list of prediction results and the metadata items
        results: List[int] = []
        metadata_items: List[IPredictionMetaData] = []

        # Initialize the lookback df
        lookback: DataFrame = Candlestick.get_lookback_df(self.max_lookback, current_timestamp)

        # Iterate over each sub model
        for model in self.sub_models:
            # Generate the prediction
            pred: IPrediction = model.predict(current_timestamp, lookback_df=lookback, enable_cache=True)

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
        """Dumps the model's data into a dictionary that will be used
        to get the insights based on its performance.

        Args:
            None

        Returns:
            IModel
        """
        # Initialize the partial model
        model: IModel = {
            "id": self.id,
            "consensus_model": {
                "sub_models": [],
                "interpreter": self.interpreter.get_config()
            }
        }

        # Iterate over each of the models and add the configurations accordingly
        for sub_model in self.sub_models:
            # Build the model's config
            config: IModel = sub_model.get_model()
            
            # Append it to the consensus model configuration
            model["consensus_model"]["sub_models"].append(config)

            # Append it to the ArimaModel configs if applies
            if isinstance(sub_model, ArimaModel):
                if isinstance(model.get("arima_models"), list):
                    model["arima_models"].append(config["arima_models"][0])
                else:
                    model["arima_models"] = config["arima_models"]

            # Append it to the RegressionModel configs if applies
            elif isinstance(sub_model, RegressionModel):
                if isinstance(model.get("regression_models"), list):
                    model["regression_models"].append(config["regression_models"][0])
                else:
                    model["regression_models"] = config["regression_models"]

            # Append it to the ClassificationModel configs if applies
            elif isinstance(sub_model, ClassificationModel):
                if isinstance(model.get("classification_models"), list):
                    model["classification_models"].append(config["classification_models"][0])
                else:
                    model["classification_models"] = config["classification_models"]

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
        return isinstance(model.get("consensus_model"), dict) and (
            isinstance(model.get("arima_models"), list) or
            isinstance(model.get("regression_models"), list) or
            isinstance(model.get("classification_models"), list)
        )