from ..keras_models.types import IKerasPath, IKerasModelConfig, IKerasModelTrainingHistory, IKerasModelOptimizerConfig, \
    IKerasModelLossConfig, IKerasModelLayer, IKerasModelSummary
from ..keras_models.KerasModelValidation import validate
from ..keras_models.KerasModelSummary import get_summary
from ..keras_models.KerasModel import KerasModel




## Keras Path Dict ##
KERAS_PATH: IKerasPath = {
    # Keras Assets
    # The root path for the assets
    "assets": "keras_assets",

    # Keras Models
    # The path in which all regression and classification models are stored
    "models": "keras_assets/models",

    # Classification Training Data
    # The path containing all the classification training data files.
    "classification_training_data": "keras_assets/classification_training_data",

    # Batched Training Certificates
    # Even though individual certificates are stored within the model's directory,
    # a batch is also saved on a different directory so multiple configurations can
    # be evaluated simultaneously.
    "batched_training_certificates": "keras_assets/batched_training_certificates",

    # Regression Training Configurations
    "regression_training_configs": "keras_assets/regression_training_configs",

    # Classification Training Data Configs
    "classification_training_data_configs": "keras_assets/classification_training_data_configs",

    # Classification Training Configs
    "classification_training_configs": "keras_assets/classification_training_configs"
}