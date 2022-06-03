from typing import List, Union
from json import loads
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from modules.model import IModel, IClassificationConfig
from modules.keras_models import KerasModelInterface, KERAS_PATH, get_summary



class Classification(KerasModelInterface):
    """Classification Class

    This class handles the initialization of a Keras Classification Model.

    Class Properties:
        ...

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        training_data_id: str
            The ID of the training data that was used to train the model.
        regressions: List[IModel]
            The list of regression models that will output the features.
        model: Sequential
            The instance of the trained model.
    """




    def __init__(self, id: str):
        """Initializes the Classification Instance.

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
        with h5pyFile(f"{KERAS_PATH['models']}/{id}/model.h5", mode='r') as model_file:
            self.id: str = model_file.attrs['id']
            self.description: str = model_file.attrs['description']
            self.training_data_id: str = model_file.attrs['training_data_id']
            self.regressions: List[IModel] = loads(model_file.attrs['models'])
            self.model: Sequential = load_model_from_hdf5(model_file)

        # Make sure the IDs are identical
        if self.id != id:
            raise ValueError(f"ClassificationModel ID Missmatch: {self.id} != {id}")

        # Make sure the description was extracted
        if not isinstance(self.description, str):
            raise ValueError(f"ClassificationModel Description is invalid: {str(self.description)}")
        
        # Make sure the training_data_id was extracted
        if not isinstance(self.training_data_id, str):
            raise ValueError(f"ClassificationModel Training Data ID is invalid: {self.training_data_id}")

        # Make sure the regressions were extracted and initialized
        if not isinstance(self.regressions, list) or len(self.regressions) < 5:
            raise ValueError(f"ClassificationModel Regressions are invalid.")








    def predict(self, features: List[Union[int, float]]) -> List[float]:
        """Given a list of features, it will perform a prediction.

        Args:
            features: List[Union[int, float]]
                The features to be used in order to generate predictions. These must match
                the ArimaModel|RegressionModels registered in the Classification.

        Returns:
            List[float]
            [up_probability, down_probability]
        """
        return self.model.predict([features])[0]











    def get_config(self) -> IClassificationConfig:
        """Returns the configuration of the Keras Classification Model.

        Returns:
            IClassificationConfig
        """
        return {
            "id": self.id,
            "description": self.description,
            "training_data_id": self.training_data_id,
            "models": self.regressions,
            "summary": get_summary(self.model),
        }