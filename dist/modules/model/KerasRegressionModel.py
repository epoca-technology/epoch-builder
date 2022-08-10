from modules._types import IModel
from modules.model.ModelType import validate_id
from modules.keras_regression.KerasRegression import KerasRegression
from modules.model.RegressionModel import RegressionModel






# Class
class KerasRegressionModel(RegressionModel):
    """KerasRegressionModel Class
    
    This class is responsible of handling interactions with Keras Regression Models.

    Instance Properties:
        ...
    """





    def __init__(self, config: IModel, enable_cache: bool = False):
        """Initializes the instance properties, the regression model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
            enable_cache: bool
                If enabled, the model will store predictions and features in the db.
        """
        # Make sure there is 1 Regression Model
        if len(config["keras_regressions"]) != 1:
            raise ValueError(f"A KerasRegressionModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['keras_regressions'])}")

        # Initialize the ID of the model
        validate_id("KerasRegressionModel", config["id"])

        # Initialize the regression
        regression: KerasRegression = KerasRegression(config["keras_regressions"][0]["regression_id"])

        # Initialize the Regression Instance
        super().__init__(config, regression, enable_cache)








    def get_model(self) -> IModel:
        """Dumps the model's data into a dictionary.

        Args:
            None

        Returns:
            IModel
        """
        return {
            "id": self.id,
            "keras_regressions": [{
                "regression_id": self.regression.id,
                "interpreter": self.interpreter.get_config(),
                "regression": self.regression.get_config()
            }]
        }







    @staticmethod
    def is_config(model: IModel) -> bool:
        """Verifies if a model is a KerasRegressionModel.

        Args:
            model: IModel
                A model configuration dict.

        Returns:
            bool
        """
        return isinstance(model.get("keras_regressions"), list) \
                and len(model["keras_regressions"]) == 1 \
                    and model.get("xgb_regressions") == None \
                        and model.get("keras_classifications") == None \
                            and model.get("xgb_classifications") == None \
                                and model.get("consensus") == None