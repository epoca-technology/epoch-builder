from typing import List
from json import loads
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from modules._types import IModel, IKerasClassificationConfig, KerasModelInterface
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
            The list of regression models that will output the features.
        include_rsi: bool
        include_aroon: bool
            Optional Technical Analysis Features
        features_num: int
            The total number of features that will be used by the model to predict
        model: Sequential
            The instance of the trained model.
    """




    def __init__(self, id: str):
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
        # Load the model
        model_path: str = Epoch.FILE.get_active_model_path(id, "keras_classification")
        with h5pyFile(model_path, mode='r') as model_file:
            self.id: str = model_file.attrs['id']
            self.description: str = model_file.attrs["description"]
            self.training_data_id: str = model_file.attrs["training_data_id"]
            self.include_rsi: bool = model_file.attrs.get("include_rsi") == True
            self.include_aroon: bool = model_file.attrs.get("include_aroon") == True
            self.features_num: int = int(model_file.attrs["features_num"]) # Downcast to int
            self.regressions: List[IModel] = loads(model_file.attrs["models"])
            self.model: Sequential = load_model_from_hdf5(model_file)
        
        # Make sure the IDs are identical
        if self.id != id:
            raise ValueError(f"KerasClassificationModel ID Missmatch: {self.id} != {id}")

        # Make sure the description was extracted
        if not isinstance(self.description, str):
            raise ValueError(f"KerasClassificationModel Description is invalid: {str(self.description)}")
        
        # Make sure the training_data_id was extracted
        if not isinstance(self.training_data_id, str):
            raise ValueError(f"KerasClassificationModel Training Data ID is invalid: {self.training_data_id}")
        
        # Make sure the features_num was extracted
        if not isinstance(self.features_num, int):
            raise ValueError(f"KerasClassificationModel features_num is invalid: {self.features_num}")

        # Make sure the regressions were extracted and initialized
        if not isinstance(self.regressions, list) or len(self.regressions) < 1:
            print(self.regressions)
            raise ValueError(f"KerasClassificationModel Regressions are invalid.")








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
            "models": self.regressions,
            "include_rsi": self.include_rsi,
            "include_aroon": self.include_aroon,
            "features_num": self.features_num,
            "summary": get_summary(self.model),
        }