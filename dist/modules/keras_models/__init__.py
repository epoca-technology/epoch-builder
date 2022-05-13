from ..keras_models.types import IKerasModelConfig, IKerasModelTrainingHistory, IKerasModelOptimizerConfig, \
    IKerasModelLossConfig, IKerasModelLayer, IKerasModelSummary
from ..keras_models.KerasModelValidation import validate
from ..keras_models.KerasModelSummary import get_summary
from ..keras_models.KerasModel import KerasModel




# Keras Models Path
# The path keras models should be saved in.
KERAS_MODELS_PATH: str = "saved_keras_models"



# Keras Models Certificates Path
# The path in which certificate batches should be stored.
KERAS_MODELS_CERTIFICATES_PATH: str = "saved_keras_models_certificates"