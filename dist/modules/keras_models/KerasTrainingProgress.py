from tqdm import tqdm
from keras.callbacks import Callback
from modules._types import IKerasModelTrainingHistory
from modules.utils.Utils import Utils



# Progress Bar embedded into Keras Callbacks
class KerasTrainingProgressBar(Callback):
    # Init
    def __init__(self, max_epochs: int, progress_bar_description: str):
        self.max_epochs: int = max_epochs
        self.progress_bar: tqdm = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=max_epochs)
        self.progress_bar.set_description(progress_bar_description)


    # Param Setter
    def set_params(self, params):
        params["max_epochs"] = 0


    # On Epoch Begin Event
    def on_epoch_begin(self, epoch: int, logs=None):
        self.progress_bar.update()






# Training Result
# Any model that does not complete at least 30% of its training is considered to
# have failed.
def training_passed(history: IKerasModelTrainingHistory, epochs: int) -> bool:
    """Verifies if a model completed at least 30% of its training.

    Args:
        history: IKerasModelTrainingHistory
            The full training history object.
        epochs: int
            The maximum number of epochs a model can go through.

    Returns bool
    """
    return Utils.get_percentage_out_of_total(len(history["loss"]), epochs) >= 30