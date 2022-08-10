from modules._types import IModel
from modules.model.ModelType import validate_id
from modules.keras_classification.KerasClassification import KerasClassification
from modules.model.ClassificationModel import ClassificationModel





# Class
class KerasClassificationModel(ClassificationModel):
    """KerasClassificationModel Class
    
    This class is responsible of handling interactions with Keras Classification Models.

    Instance Properties:
        ...
    """





    def __init__(self, config: IModel, enable_cache: bool = False):
        """Initializes the instance properties, the classification model and the Interpreter's Instance.

        Args:
            config: IModel
                The configuration to be used to initialize the model's instance
            enable_cache: bool
                If enabled, the model will store predictions in the db. Keep in mind that the regressions
                within the classification use cache by default.
        """
        # Make sure there is 1 Classification Model
        if len(config["keras_classifications"]) != 1:
            raise ValueError(f"A KerasClassificationModel can only be initialized if 1 configuration item is provided. \
                Received: {len(config['keras_classifications'])}")

        # Initialize the ID of the model
        validate_id("KerasClassificationModel", config["id"])

        # Initialize the classification
        classification: KerasClassification = KerasClassification(config["keras_classifications"][0]["classification_id"])

        # Initialize the Regression Instance
        super().__init__(config, classification, enable_cache)







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