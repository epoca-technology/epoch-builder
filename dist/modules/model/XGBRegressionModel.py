from modules._types import IModel
from modules.model.ModelType import validate_id
from modules.xgb_regression.XGBRegression import XGBRegression
from modules.model.RegressionModel import RegressionModel





class XGBRegressionModel(RegressionModel):
    """XGBRegressionModel Class
    
    This class is responsible of handling interactions with XGBoost Regression Models.

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
        if len(config["xgb_regressions"]) != 1:
            raise ValueError(f"A XGBRegressionModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['xgb_regressions'])}")

        # Initialize the ID of the model
        validate_id("XGBRegressionModel", config["id"])

        # Initialize the regression
        regression: XGBRegression = XGBRegression(config["xgb_regressions"][0]["regression_id"])

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