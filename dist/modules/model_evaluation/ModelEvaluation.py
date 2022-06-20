from typing import Union, List
from pandas import DataFrame
from numpy import mean
from tqdm import tqdm
from modules.types import IModel, IPrediction, IBacktestPerformance, IModelEvaluation
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.model.RegressionModel import RegressionModel
from modules.model.ClassificationModel import ClassificationModel
from modules.model.ModelFactory import ModelFactory
from modules.backtest.Position import Position




def evaluate(
    model_config: IModel,
    start_timestamp: int,
    price_change_requirement: float,
    hyperparams_mode: bool
) -> IModelEvaluation:
    """Performs an evaluation on a recently trained model. The evaluation works 
    similarly to the backtests and it is only performed on the test dataset to 
    ensure the model has not yet seen it.

    Args:
        model: IModel
            The model to be initialized and evaluated.
        start_timestamp: int
            The open time of the first candlestick in the test dataset.
        price_change_requirement: float
            The price percentage change requirement in order for an outcome to be 
            determined. F.e. If increase_requirement is 3 and the price goes
            up 3%, the outcome will be "increased".
        hyperparams_mode: bool
            The type of training being performed. If provided, enables some verbose
            features.

    Returns:
        IModelEvaluation
    """
    # Initialize the model
    model: Union[RegressionModel, ClassificationModel] = ModelFactory(model_config)

    # Initialize the type of model
    model_type: str = type(model).__name__

    # Init the test dataset
    df: DataFrame = Candlestick.DF[Candlestick.DF['ot'] >= start_timestamp]
    df.reset_index(drop=True, inplace=True)

    # Init evaluation data
    neutral_predictions: int = 0
    increase: List[float] = []
    increase_successful: List[float] = []
    decrease: List[float] = []
    decrease_successful: List[float] = []
    increase_outcomes: int = 0
    decrease_outcomes: int = 0

    # Init the progress bar
    if not hyperparams_mode:
        progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=df.shape[0])
        progress_bar.set_description(f"    6/7) Evaluating {model_type}")

    # Init the Position Instance
    position: Position = Position(price_change_requirement, price_change_requirement)

    # Last neutral prediction range close time.
    last_neutral_ct: int = 0

    # Iterate over each 1 minute candlestick from the test dataset
    for candlestick_index, candlestick in df.iterrows():
        # Check if it is the last candlestick
        is_last_candlestick: bool = candlestick_index == df.index[-1]

        # Active Position
        # Check the active position against the new candlestick. If the position is closed,
        # process the metrics
        if position.active != None:
            # Check the position against the new candlestick
            closed_position: bool = position.check_position(candlestick)

            # If the position has been closed, process the metrics
            if closed_position:
                # Extract the metadata value
                metadata_value: float = _get_prediction_metadata_value(model_type, position.positions[-1]["p"])

                # Handle an increase prediction
                if position.positions[-1]["t"] == 1:
                    # Append the increase prediction to the list
                    increase.append(metadata_value)

                    # Check if the prediction was correct
                    if position.positions[-1]["o"]:
                        increase_successful.append(metadata_value)
                        increase_outcomes += 1
                    else:
                        decrease_outcomes += 1

                # Handle a decrease prediction
                else:
                    # Append the decrease prediction to the list
                    decrease.append(metadata_value)
                    
                    # Check if the prediction was correct
                    if position.positions[-1]["o"]:
                        decrease_successful.append(metadata_value)
                        decrease_outcomes += 1
                    else:
                        increase_outcomes += 1

        # Inactive Position
        # A new prediction will be generated if the following is met:
        # 1) There isn't an active position
        # 2) It isn't the last candlestick
        # 3) The current prediction range's close time is greater than the last one
        elif (position.active == None) and (not is_last_candlestick):
            # Retrieve the current prediction range's close time
            _, last_ct = Candlestick.get_lookback_prediction_range(100, candlestick["ot"])

            # Only predict in new ranges
            if last_ct > last_neutral_ct:
                # Generate a prediction
                pred: IPrediction = model.predict(candlestick['ot'], enable_cache=False)

                # If the result isn't neutral, open a position
                if pred['r'] != 0:
                    # Downcast the probability values if applies, so the JSON file can be saved
                    if model_type == "ClassificationModel":
                        pred["md"][0]["up"] = float(pred["md"][0]["up"])
                        pred["md"][0]["dp"] = float(pred["md"][0]["dp"])
                    
                    # Open a position
                    position.open_position(candlestick, pred)

                # Otherwise, handle the neutrality
                else:
                    # Increase the neutral counter
                    neutral_predictions += 1

                    # Set the last_ct in the current range
                    last_neutral_ct = last_ct

        # Update the progress bar
        if not hyperparams_mode:
            progress_bar.update()

    # Output the performance
    performance: IBacktestPerformance = position.get_performance()

    # Initialize the position type lengths
    increase_successful_num: int = len(increase_successful)
    decrease_successful_num: int = len(decrease_successful)

    # Finally, return the results
    return {
        # Neutral Predictions
        "neutral_predictions": neutral_predictions,

        # Positions
        "positions": performance["positions"],

        # Prediction counts
        "increase_num": performance["long_num"],
        "increase_successful_num": increase_successful_num,
        "decrease_num": performance["short_num"],
        "decrease_successful_num": decrease_successful_num,

        # Accuracy
        "increase_acc": performance["long_acc"],
        "decrease_acc": performance["short_acc"],
        "acc": performance["general_acc"],
        
        # Predictions Overview 
        "increase_list": increase,
        "increase_max": max(increase if performance["long_num"] > 0 else [0]),
        "increase_min": min(increase if performance["long_num"] > 0 else [0]),
        "increase_mean": mean(increase if performance["long_num"] > 0 else [0]),
        "increase_successful_list": increase_successful,
        "increase_successful_max": max(increase_successful if increase_successful_num > 0 else [0]),
        "increase_successful_min": min(increase_successful if increase_successful_num > 0 else [0]),
        "increase_successful_mean": mean(increase_successful if increase_successful_num > 0 else [0]),
        "decrease_list": decrease,
        "decrease_max": max(decrease if performance["short_num"] > 0 else [0]),
        "decrease_min": min(decrease if performance["short_num"] > 0 else [0]),
        "decrease_mean": mean(decrease if performance["short_num"] > 0 else [0]),
        "decrease_successful_list": decrease_successful,
        "decrease_successful_max": max(decrease_successful if decrease_successful_num > 0 else [0]),
        "decrease_successful_min": min(decrease_successful if decrease_successful_num > 0 else [0]),
        "decrease_successful_mean": mean(decrease_successful if decrease_successful_num > 0 else [0]),

        # Outcomes
        "increase_outcomes": increase_outcomes,
        "decrease_outcomes": decrease_outcomes,
    }









def _get_prediction_metadata_value(model_type: str, pred: IPrediction) -> float:
    """Extracts the metadata value for a given prediction based on the type of model.

    Args:
        model_type: str
            The type of model being evaluated.
        pred: IPrediction
            The prediction generated by the model.

    Returns:
        float
    """
    # Extract the data from a Regression Model
    if model_type == "RegressionModel":
        return Utils.get_percentage_change(pred["md"][0]["npl"][0], pred["md"][0]["npl"][-1])

    # Extract the data from a Classification Model
    elif model_type == "ClassificationModel":
        if pred["r"] == 1:
            return float(pred["md"][0]["up"])
        else:
            return float(pred["md"][0]["dp"])

    # Otherwise, stop the execution
    else:
        raise ValueError(f"Cannot extract the metadata value from an invalid model type {model_type}")