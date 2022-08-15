from typing import List, Tuple, Union
from numpy import mean
from pandas import DataFrame, Series
from modules._types import IEarlyStoppingProcessName, IDiscovery, IDiscoveryPayload, IPositionType
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.process_early_stopping.ProcessEarlyStopping import ProcessEarlyStopping



# Training Failed Early Stopping
TRAINING_FAILED_ES: str = "The discovery was skipped because the training was stopped too early."




def get_real_steps() -> int:
    """Returns the number of 1 minute candlesticks that are contained in 1 
    step.

    Returns:
        int
    """
    return Epoch.MODEL_DISCOVERY_STEPS * Candlestick.PREDICTION_CANDLESTICK_CONFIG["interval_minutes"]






def get_candlesticks_df(lookback: int) -> DataFrame:
    """The models need data prior to the current time to perform predictions. Since the default candlesticks
    will be used for simulating, the df needs to start from a point in which there are enough prediction
    candlesticks in order to make a prediction. Once the subsetting is done, reset the indexes.

    Args:
        lookback: int
            The number of prediction candlesticks the model requires in order to make
            a prediction.

    Returns:
        DataFrame
    """
    # Calculate the real start time based on the model's lookback
    real_start: int = Candlestick.PREDICTION_DF[Candlestick.PREDICTION_DF["ot"] >= Epoch.BACKTEST_START].iloc[lookback]["ot"]

    # Init 1m candlesticks dataframe
    df: DataFrame = Candlestick.DF[Candlestick.DF["ot"] >= real_start]
    df.reset_index(drop=True, inplace=True)

    # Finally, return the 1m candlesticks df
    return df








def get_early_stopping(process_name: IEarlyStoppingProcessName, candlesticks_num: int) -> ProcessEarlyStopping:
    """Retrieves the Early Stopping instance that will be used for discoveries.

    Args:
        process_name: IProcessName
            The name of the process.
        candlesticks_num: int
            The number of 1 minute candlesticks within the process.

    Returns:
        ProcessEarlyStopping
    """
    return ProcessEarlyStopping(
        process_name=process_name,
        candlesticks_num=candlesticks_num,
        checkpoints=[
            { "required_longs": 3, "required_shorts": 3, "dataset_percent": 0.15 },
            { "required_longs": 5, "required_shorts": 5, "dataset_percent": 0.3 },
            { "required_longs": 10, "required_shorts": 10, "dataset_percent": 0.5 },
            { "required_longs": 15, "required_shorts": 15, "dataset_percent": 0.7 },
        ]
    )






def calculate_exit_prices(open_price: float, price_change_requirement: float) -> Tuple[float, float]:
    """Calculates the take profit and the stop loss prices for a  discovery position 
    about to be opened.

    Args:
        open_price: float
            The current candlestick's open price.
        price_change_requirement: float
            The percentage the price needs to change to claim profit or cut losses.
    
    Returns:
        Tuple[float, float] 
        (take_profit, stop_loss)
    """
    return Utils.alter_number_by_percentage(open_price, price_change_requirement), \
        Utils.alter_number_by_percentage(open_price, -(price_change_requirement))







def check_position(position_type: IPositionType, candlestick: Series, take_profit: float, stop_loss: float) -> Union[float, None]:
    """Checks if the current 1 minute candlestick has hit the take profit or the stop loss.
    Returns None if neither of the prices has been hit

    Args:
        position_type: IPositionType
            The type of position.
        candlestick: Series
            The current 1 minute candlestick
        take_profit: float
        stop_loss: float
            Exit Prices
    
    Returns:
        Union[float, None]
    """
    # Check if it is a long position
    if position_type == 1:
        # Check if the stop loss has been hit by the low
        if candlestick["l"] <= stop_loss:
            return stop_loss

        # Check if the take profit has been hit by the high
        elif candlestick["h"] >= take_profit:
            return take_profit

    # Check if the short needs to be closed
    else:
        # Check if the stop loss has been hit by the high
        if candlestick["h"] >= stop_loss:
            return stop_loss

        # Check if the take profit has been hit by the low
        elif candlestick["l"] <= take_profit:
            return take_profit

    # Otherwise, the position remains active
    return None







def build_discovery(
    early_stopping: Union[str, None],
    neutral_num: int,
    increase_num: int,
    increase: List[float],
    increase_successful_num: int,
    increase_successful: List[float],
    decrease_num: int,
    decrease: List[float],
    decrease_successful_num: int,
    decrease_successful: List[float]
) -> Tuple[IDiscovery, IDiscoveryPayload]:
    """Given the raw data, it will build the discovery that will be attached to the
    model and the payload that will be attached to the certificate.

    Args:
        early_stopping: Union[str, None]
        neutral_num: int
        increase_num: int
        increase: List[float]
        increase_successful_num: int
        increase_successful: List[float]
        decrease_num: int
        decrease: List[float]
        decrease_successful_num: int
        decrease_successful: List[float]

    Returns:
        Tuple[IDiscovery, IDiscoveryPayload]
    """
    # Init helpers
    increase_successful_mean = round(mean(increase_successful), 2)
    decrease_successful_mean = round(mean(decrease_successful), 2)

    # Calculate the total non neutral predictions
    non_neutral_num: int = increase_num + decrease_num

    # Calculate the total successful predictions
    successful_num: int = increase_successful_num + decrease_successful_num
    
    # Build the discovery payload
    payload: IDiscoveryPayload = {
        # Early Stopping Motive (If any)
        "early_stopping": early_stopping,

        # Position & Outcome Counts
        "neutral_num": neutral_num,
        "increase_num": increase_num,
        "decrease_num": decrease_num,
        "increase_outcome_num": increase_successful_num + (decrease_num - decrease_successful_num),
        "decrease_outcome_num": decrease_successful_num + (increase_num - increase_successful_num),

        # Accuracies
        "increase_accuracy": Utils.get_percentage_out_of_total(
            increase_successful_num, increase_num if increase_num > 0 else 1
        ),
        "decrease_accuracy": Utils.get_percentage_out_of_total(
            decrease_successful_num, decrease_num if decrease_num > 0 else 1
        ),
        "accuracy": Utils.get_percentage_out_of_total(
            successful_num, non_neutral_num if non_neutral_num > 0 else 1
        ),

        # Increase Predictions Details
        "increase_list": increase,
        "increase_max": round(max(increase if increase_num > 0 else [0]), 2),
        "increase_min": round(min(increase if increase_num > 0 else [0]), 2),
        "increase_mean": round(mean(increase if increase_num > 0 else [0]), 2),

        # Decrease Predictions Details
        "decrease_list": decrease,
        "decrease_max": round(max(decrease if decrease_num > 0 else [0]), 2),
        "decrease_min": round(min(decrease if decrease_num > 0 else [0]), 2),
        "decrease_mean": round(mean(decrease if decrease_num > 0 else [0]), 2),

        # Details of the successful increase predictions
        "increase_successful_list": increase_successful,
        "increase_successful_mean": increase_successful_mean,

        # Details of the successful decrease predictions
        "decrease_successful_list": decrease_successful,
        "decrease_successful_mean": decrease_successful_mean,

        # The mean of the successful increase and decrease means
        "successful_mean": _calculate_successful_mean(increase_successful_mean, decrease_successful_mean)
    }

    # Build the discovery
    discovery: IDiscovery = {
        # Details of the increase predictions
        "increase_min": payload["increase_min"],
        "increase_max": payload["increase_max"],

        # Details of the decrease predictions
        "decrease_min": payload["decrease_min"],
        "decrease_max": payload["decrease_max"],

        # Details of the successful predictions
        "increase_successful_mean": payload["increase_successful_mean"],
        "decrease_successful_mean": payload["decrease_successful_mean"],

        # The mean of the successful increase and decrease means
        # This value is used by the RegressionSelection module to calculate the optimal 
        # price_change_requirement for Classifications.
        # Even though this value is present in ClassificationDiscovery, it is not actually
        # used.
        "successful_mean": payload["successful_mean"],
    }

    # Finally, pack and return the values
    return discovery, payload





def _calculate_successful_mean(increase_successful_mean: float, decrease_successful_mean: float) -> float:
    """In the case of regressions, the decrease_successful_mean will be a negative number,
    therefore, it must be converted to positive prior to calculating the mean.

    Args:
        increase_successful_mean: float
        decrease_successful_mean: float
            Price changes or probabilities generated by the model.
    
    Returns:
        float
    """
    # Calculate the absolute value of the decrease_successful_mean
    absolute_decrease_successful_mean: float = \
        decrease_successful_mean if decrease_successful_mean > 0 else -(decrease_successful_mean)

    # Calculate the mean of both values
    return round(mean([increase_successful_mean, absolute_decrease_successful_mean]), 2)