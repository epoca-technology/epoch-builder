from typing import List, Tuple, Union
from math import ceil
from pandas import DataFrame
from tqdm import tqdm
from modules._types import IDiscovery, IDiscoveryPayload
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.keras_regression.KerasRegression import KerasRegression
from modules.xgb_regression.XGBRegression import XGBRegression
from modules.process_early_stopping.ProcessEarlyStopping import ProcessEarlyStopping
from modules.discovery.Discovery import get_real_steps, get_candlesticks_df, get_early_stopping, calculate_exit_prices,\
    check_position, build_discovery



# Min and max increase changes allowed
MIN_INCREASE: float = 1
MAX_INCREASE: float = 4

# Min and max decrease changes allowed
MIN_DECREASE: float = -4
MAX_DECREASE: float = -1






def discover(regression: Union[KerasRegression, XGBRegression], progress_bar_description: str) -> Tuple[IDiscovery, IDiscoveryPayload]:
    """Performs a discovery on a regression in order to find its optimal parameters 
    prior to evaluating it. If the model shows poor performance, the discovery
    process will be stopped and the motive will be provided in the 
    discovery payload.

    Args:
        regression: Union[KerasRegression, XGBRegression]
            The regression to be discovered.
        progress_bar_description: str
            The description to be set on the progress bar.

    Returns:
        Tuple[IDiscovery, IDiscoveryPayload]
    """

    # Retrieve the Candlesticks Dataframe
    df: DataFrame = get_candlesticks_df(regression.lookback)

    # Calculate the number of 1 minute candlesticks that will be stepped
    real_steps: int = get_real_steps()

    # Init the progress bar
    progress_bar = tqdm(bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=ceil(df.shape[0]/real_steps))
    progress_bar.set_description(progress_bar_description)

    # Early Stopping
    # Init the values that will help evaluate if the regression's discovery should continue 
    # or be aborted in order to save time and resources.
    es: ProcessEarlyStopping = get_early_stopping(process_name="RegressionDiscovery", candlesticks_num=df.shape[0])
    es_motive: Union[str, None] = None

    # Init discovery data
    neutral_num: int = 0
    increase_num: int = 0
    increase: List[float] = []
    increase_successful: List[float] = []
    decrease_num: int = 0
    decrease: List[float] = []
    decrease_successful: List[float] = []

    # Iterate over the candlesticks based on the real steps
    for index in range(0, df.shape[0]-1, real_steps):
        # Generate a prediction
        change: float = _predict_change(df.iloc[index]["ot"], regression)

        # Check if it is a non-neutral prediction
        if change != 0:
            # Init the position exit prices
            take_profit, stop_loss = calculate_exit_prices(df.iloc[index]["o"], change)
            
            # Iterate until the position's outcome has been determined
            outcome_index: int = index + 1
            close_price: Union[float, None] = None
            while close_price == None and outcome_index < df.shape[0]:
                # Check the position
                close_price = check_position(1 if change > 0 else -1, df.iloc[outcome_index], take_profit, stop_loss)

                # Increment the index
                outcome_index += 1

            # Check if the model predicted an increase
            if change > 0:
                # Process increase prediction
                increase_num += 1
                increase.append(change)

                # Check if it was successful
                if take_profit == close_price:
                    increase_successful.append(change)

            # Check if the model predicted a decrease
            else:
                # Process decrease prediction
                decrease_num += 1
                decrease.append(change)

                # Check if it was successful
                if take_profit == close_price:
                    decrease_successful.append(change)

        # Otherwise, handle the neutral prediction
        else:
            neutral_num += 1

        # Perform the early stopping evaluation
        es_motive = es.check(index, increase_num, decrease_num)

        # Check if the early stopping has been triggered
        if isinstance(es_motive, str):
            break

        # Otherwise, update the progress bar and move on to the next candlestick
        else:
            progress_bar.update()

    # Init the successful prediction counts
    increase_successful_num: int = len(increase_successful)
    decrease_successful_num: int = len(decrease_successful)

    # Finally, return the discovery and the payload
    return build_discovery(
        early_stopping=es_motive,
        neutral_num=neutral_num,
        increase_num=increase_num,
        increase=increase,
        increase_successful_num=increase_successful_num,
        increase_successful=increase_successful if increase_successful_num > 0 else [MIN_INCREASE],
        decrease_num= decrease_num,
        decrease=decrease,
        decrease_successful_num=decrease_successful_num,
        decrease_successful=decrease_successful if decrease_successful_num > 0 else [MAX_DECREASE]
    )







def _predict_change(current_timestamp: int, regression: Union[KerasRegression, XGBRegression]) -> float:
    """Performs a regression prediction and returns the adjusted change.

    Args:
        current_timestamp: int
            The open time of the current 1 minute candlestick.
        regression: Union[KerasRegression, XGBRegression]
            The regression being discovered.

    Returns:
        float
    """
    # Retrieve the normalized lookback df
    df: DataFrame = Candlestick.get_lookback_df(regression.lookback, current_timestamp, normalized=True)

    # Generate the predictions
    preds: List[float] = regression.predict(df["c"])

    # Calculate the change between the current price and the last prediction
    change: float = Utils.get_percentage_change(df["c"].iloc[-1], preds[-1])
    
    # Return the adjusted change
    if change >= MIN_INCREASE and change < MAX_INCREASE:
        return change
    elif change >= MAX_INCREASE:
        return MAX_INCREASE
    elif change > MIN_DECREASE and change <= MAX_DECREASE:
        return change
    elif change <= MIN_DECREASE:
        return MIN_DECREASE
    else:
        return 0