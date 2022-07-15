from typing import List
from tqdm import tqdm
from modules.types import IPredictionRangeIndexer
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.EpochFile import EpochFile





def create_indexer(lookbacks: List[int], progress_bar_description: str) -> None:
    """Creates the lookback prediction range indexer based on the 
    candlestick bundle.

    Args:
        lookbacks: List[int]
            The lookbacks that will be used when indexing
        progress_bar_description: str
            The description to be set on the progress bar.
    """
    # Initialize the indexer that will hold the values
    indexer: IPredictionRangeIndexer = {}

    # Init the progress bar
    progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=len(lookbacks)*Candlestick.DF.shape[0])
    progress_bar.set_description(progress_bar_description)

    # Iterate over each lookback
    for lookback in lookbacks:
        # Iterate over each candlestick
        for _, candlestick in Candlestick.DF.iterrows():
            # Init the ID of the range
            id: str = Candlestick._get_lookback_prediction_range_id(lookback, candlestick["ot"])

            # Calculate the range
            first_ot, last_ct = Candlestick._calculate_lookback_prediction_range(lookback, candlestick["ot"])

            # Populate the indexer
            indexer[id] = [first_ot, last_ct]

            # Update the progress bar
            progress_bar.update()

    # Finally, save the indexer
    EpochFile.write(Candlestick.PREDICTION_RANGE_INDEXER_PATH, indexer)