from typing import List, Union
from json import loads
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from modules._types import IModel, IKerasClassificationConfig, KerasModelInterface, IDiscovery,\
    IKerasClassificationDiscoveryInitConfig
from modules.epoch.Epoch import Epoch
from modules.keras_models.KerasModelSummary import get_summary



class KerasClassification(KerasModelInterface):
    """KerasClassification Class

    This class handles the initialization of a Keras Classification Model.

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        training_data_id: str
            The ID of the training data that was used to train the model.
        regressions: List[IModel]
            The list of regression regressions that will output the features.
        include_rsi: bool
        include_aroon: bool
            Optional Technical Analysis Features
        features_num: int
            The total number of features that will be used by the model to predict
        price_change_requirement: float
            The best price change extracted from the RegressionSelection.
        discovery: IDiscovery
            The model's discovery information. If the model has not yet been saved, this
            value will be an empty dict.
        model: Sequential
            The instance of the trained model.
    """




    def __init__(self, id: str, discovery_config: Union[IKerasClassificationDiscoveryInitConfig, None] = None):
        """Initializes the KerasClassification Instance.

        Args:
            id: str
                The ID of the model that will be initialized.

        Raises:
            ValueError:
                If there is an issue loading the model.
                If the ID stored in the model's file is different to the one provided.
                If any of the other metadata is invalid.
        """
        # If the configuration was provided, use its values
        if isinstance(discovery_config, dict):
            self.id: str = id
            self.description: str = "Model initialized in discovery mode."
            self.training_data_id: str = discovery_config["training_data_id"]
            self.include_rsi: bool = discovery_config["include_rsi"]
            self.include_aroon: bool = discovery_config["include_aroon"]
            self.features_num: int = discovery_config["features_num"]
            self.regressions: List[IModel] = discovery_config["regressions"]
            self.price_change_requirement: float = discovery_config["price_change_requirement"]
            self.discovery: IDiscovery = {}
            self.model: Sequential = discovery_config["model"]

        # Otherwise, load the file
        else:
            Epoch.FILE.activate_model(id)
            with h5pyFile(Epoch.FILE.get_active_model_path(id, "keras_classification"), mode='r') as model_file:
                self.id: str = model_file.attrs['id']
                self.description: str = model_file.attrs["description"]
                self.training_data_id: str = model_file.attrs["training_data_id"]
                self.include_rsi: bool = bool(model_file.attrs.get("include_rsi") == True) # Downcast to bool
                self.include_aroon: bool = bool(model_file.attrs.get("include_aroon") == True) # Downcast to bool
                self.features_num: int = int(model_file.attrs["features_num"]) # Downcast to int
                self.regressions: List[IModel] = loads(model_file.attrs["regressions"])
                self.price_change_requirement: float = float(model_file.attrs["price_change_requirement"]) # Downcast to float
                self.discovery: IDiscovery = loads(model_file.attrs["discovery"])
                self.model: Sequential = load_model_from_hdf5(model_file)
        
        # Make sure the IDs are identical
        if self.id != id:
            raise ValueError(f"KerasClassification ID Missmatch: {self.id} != {id}")

        # Make sure the description was extracted
        if not isinstance(self.description, str):
            raise ValueError(f"KerasClassification Description is invalid: {str(self.description)}")
        
        # Make sure the training_data_id was extracted
        if not isinstance(self.training_data_id, str):
            raise ValueError(f"KerasClassification Training Data ID is invalid: {self.training_data_id}")
        
        # Make sure the features_num was extracted
        if not isinstance(self.features_num, int):
            raise ValueError(f"KerasClassification features_num is invalid: {self.features_num}")

        # Make sure the regressions were extracted and initialized
        if not isinstance(self.regressions, list) or len(self.regressions) < 1:
            print(self.regressions)
            raise ValueError(f"KerasClassification Regressions are invalid.")
        
        # Make sure the price_change_requirement was extracted
        if not isinstance(self.price_change_requirement, (int, float)):
            raise ValueError(f"KerasClassification price_change_requirement is invalid: {self.price_change_requirement}")

        # Make sure the discovery was extracted
        if not isinstance(self.discovery, dict):
            raise ValueError(f"KerasClassification Discovery is invalid: {str(self.discovery)}")








    def predict(self, features: List[float]) -> List[float]:
        """Given a list of features, it will perform a prediction.

        Args:
            features: List[float]
                The features to be used in order to generate predictions. These must match
                the classification's regressions and the technical analysis features (if any)

        Returns:
            List[float]
            [up_probability, down_probability]
        """
        return self.model.predict([features])[0].tolist()











    def get_config(self) -> IKerasClassificationConfig:
        """Returns the configuration of the Keras Classification Model.

        Returns:
            IClassificationConfig
        """
        return {
            "id": self.id,
            "description": self.description,
            "training_data_id": self.training_data_id,
            "regressions": self.regressions,
            "include_rsi": self.include_rsi,
            "include_aroon": self.include_aroon,
            "features_num": self.features_num,
            "price_change_requirement": self.price_change_requirement,
            "discovery": self.discovery,
            "summary": get_summary(self.model),
        }