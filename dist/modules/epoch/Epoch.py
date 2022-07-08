from typing import List
from random import seed
from numpy.random import seed as npseed
from tensorflow import random as tf_random








## EPOCH SINGLETON ##


class Epoch:
    """Epoch Class

    This singleton manages the initialization, creation and exporting of epochs.

    Class Properties:
        SEED: int
            The seed to be set on all required libs.
    """
    # Random seed to be set on all required libraries
    SEED: int = 60184



    @staticmethod
    def init() -> None:
        """
        """
        # Set a static seed on all required libraries
        seed(Epoch.SEED)
        npseed(Epoch.SEED)
        tf_random.set_seed(Epoch.SEED)
        