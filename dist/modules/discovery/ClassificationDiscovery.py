from typing import List, Tuple, Union
from math import ceil
from pandas import DataFrame
from tqdm import tqdm
from modules._types import IDiscovery, IDiscoveryPayload, IPredictionResult
from modules.candlestick.Candlestick import Candlestick
from modules.keras_classification.KerasClassification import KerasClassification
from modules.xgb_classification.XGBClassification import XGBClassification
from modules.model.RegressionModelFactory import RegressionModelFactory, RegressionModel
from modules.model.ClassificationFeatures import build_features
from modules.process_early_stopping.ProcessEarlyStopping import ProcessEarlyStopping
from modules.discovery.Discovery import get_real_steps, get_candlesticks_df, get_early_stopping, calculate_exit_prices,\
    check_position, build_discovery



# Min and max probabilities allowed
MIN_PROB: float = 0.6
MAX_PROB: float = 0.65






def discover(classification: Union[KerasClassification, XGBClassification], progress_bar_description: str) -> Tuple[IDiscovery, IDiscoveryPayload]:
    """Performs a discovery on a classification in order to find its optimal parameters 
    prior to evaluating it. If the model shows poor performance, the discovery
    process will be stopped and the motive will be provided in the 
    discovery payload.

    Args:
        classification: Union[KerasClassification, XGBClassification]
            The classification to be discovered.
        progress_bar_description: str
            The description to be set on the progress bar.

    Returns:
        Tuple[IDiscovery, IDiscoveryPayload]
    """
    # Initialize the regression models
    regressions: List[RegressionModel] = [ RegressionModelFactory(m, True) for m in classification.regressions ]

    # Initialize the max lookback
    max_lookback: int = max([m.get_lookback() for m in regressions])

    # Retrieve the Candlesticks Dataframe
    df: DataFrame = get_candlesticks_df(max_lookback)

    # Calculate the number of 1 minute candlesticks that will be stepped
    real_steps: int = get_real_steps()

    # Init the progress bar
    progress_bar = tqdm(bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=ceil(df.shape[0]/real_steps))
    progress_bar.set_description(progress_bar_description)

    # Early Stopping
    # Init the values that will help evaluate if the regression's discovery should continue 
    # or be aborted in order to save time and resources.
    es: ProcessEarlyStopping = get_early_stopping(process_name="ClassificationDiscovery", candlesticks_num=df.shape[0])
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
        result, probability = _predict(
            df.iloc[index]["ot"], 
            classification, 
            regressions,
            max_lookback, 
            classification.include_rsi, 
            classification.include_aroon
        )

        # Check if it is a non-neutral prediction
        if result != 0:
            # Init the position exit prices
            take_profit, stop_loss = calculate_exit_prices(
                df.iloc[index]["o"], 
                classification.price_change_requirement if result == 1 else -(classification.price_change_requirement)
            )
            
            # Iterate until the position's outcome has been determined
            outcome_index: int = index + 1
            close_price: Union[float, None] = None
            while close_price == None and outcome_index < df.shape[0]:
                # Check the position
                close_price = check_position(result, df.iloc[outcome_index], take_profit, stop_loss)

                # Increment the index
                outcome_index += 1

            # Check if the model predicted an increase
            if result == 1:
                # Process increase prediction
                increase_num += 1
                increase.append(probability)

                # Check if it was successful
                if take_profit == close_price:
                    increase_successful.append(probability)

            # Check if the model predicted a decrease
            else:
                # Process decrease prediction
                decrease_num += 1
                decrease.append(probability)

                # Check if it was successful
                if take_profit == close_price:
                    decrease_successful.append(probability)

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
        increase=increase if increase_num > 0 else [MIN_PROB, MAX_PROB],
        increase_successful_num=increase_successful_num,
        increase_successful=increase_successful if increase_successful_num > 0 else [MIN_PROB, MAX_PROB],
        decrease_num= decrease_num,
        decrease=decrease if decrease_num > 0 else [MIN_PROB, MAX_PROB],
        decrease_successful_num=decrease_successful_num,
        decrease_successful=decrease_successful if decrease_successful_num > 0 else [MIN_PROB, MAX_PROB]
    )







def _predict(
    current_timestamp: int, 
    classification: Union[KerasClassification, XGBClassification],
    regressions: List[RegressionModel],
    max_lookback: int,
    include_rsi: bool,
    include_aroon: bool
) -> Tuple[IPredictionResult, float]:
    """Performs a classification prediction and returns the type of 
    prediction, as well as the adjusted probability

    Args:
        current_timestamp: int
            The open time of the current 1 minute candlestick.
        classification: Union[KerasClassification, XGBClassification]
            The classification being discovered.

    Returns:
        Tuple[IPredictionResult, float]
        (result, probability)
    """
    # Retrieve the lookback df
    df: DataFrame = Candlestick.get_lookback_df(max_lookback, current_timestamp)

    # Build the features
    features: List[float] = build_features(
        current_timestamp=current_timestamp, 
        regressions=regressions, 
        max_lookback=max_lookback, 
        include_rsi=include_rsi,
        include_aroon=include_aroon,
        lookback_df=df
    )

    # Generate a prediction based on the features
    pred: List[float] = classification.predict(features)
    
    # Return the adjusted probability
    if pred[0] >= MIN_PROB and pred[0] <= MAX_PROB:
        return 1, pred[0]
    elif pred[0] > MAX_PROB:
        return 1, MAX_PROB
    elif pred[1] >= MIN_PROB and pred[1] <= MAX_PROB:
        return -1, pred[1]
    elif pred[1] > MAX_PROB:
        return -1, MAX_PROB
    else:
        return 0