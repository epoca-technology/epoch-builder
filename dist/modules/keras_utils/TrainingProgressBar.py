from tqdm import tqdm
from keras.callbacks import Callback
from modules.utils.Utils import Utils




# Progress Bar embedded into Keras Callbacks
class TrainingProgressBar(Callback):
    # Init
    def __init__(self, active_epoch_path: str, initial_epoch: int, max_epochs: int, progress_bar_description: str):
        self.max_epochs: int = max_epochs
        self.progress_bar: tqdm = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=max_epochs)
        self.progress_bar.set_description(progress_bar_description)
        self.active_epoch_path: str = active_epoch_path
        if initial_epoch > 0:
            self.progress_bar.update(initial_epoch)


    # Param Setter
    def set_params(self, params):
        params["max_epochs"] = 0


    # On Epoch Begin Event
    def on_epoch_begin(self, epoch: int, logs=None):
        if epoch > 0:
            self.progress_bar.update()
            Utils.write(self.active_epoch_path, str(epoch))

