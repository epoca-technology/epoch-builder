from typing import Union, List
from pandas import DataFrame
from numpy import mean, median
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
    progress_bar_description: str
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
        progress_bar_description: str
            The description that will be placed in the progress bar.

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
    progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=df.shape[0])
    progress_bar.set_description(progress_bar_description)

    # Init the Position Instance
    position: Position = Position(price_change_requirement, price_change_requirement)

    # Last neutral prediction range close time.
    last_neutral_ct: int = 0

    # Early Stopping
    # Init the values that will help evaluate if the model's evaluation should continue 
    # or be aborted in order to save time and resources.
    early_stopping: Union[str, None] = None
    es_checkpoint_1: int = int(df.shape[0] * 0.15)
    es_checkpoint_2: int = int(df.shape[0] * 0.3)
    es_checkpoint_3: int = int(df.shape[0] * 0.5)
    es_checkpoint_4: int = int(df.shape[0] * 0.7)

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
                    position.open_position(candlestick, pred)

                # Otherwise, handle the neutrality
                else:
                    # Increase the neutral counter
                    neutral_predictions += 1

                    # Set the last_ct in the current range
                    last_neutral_ct = last_ct

        # Perform the early stopping evaluation
        early_stopping = _perform_early_stopping_evaluation(
            current_points=position.points[-1],
            eval_checkpoint_1=candlestick_index >= es_checkpoint_1 and candlestick_index < es_checkpoint_2,
            eval_checkpoint_2=candlestick_index >= es_checkpoint_2 and candlestick_index < es_checkpoint_3,
            eval_checkpoint_3=candlestick_index >= es_checkpoint_3 and candlestick_index < es_checkpoint_4,
            eval_checkpoint_4=candlestick_index >= es_checkpoint_4,
            longs_num=position.long_num,
            shorts_num=position.short_num
        )

        # Check if the early stopping has been triggered
        if isinstance(early_stopping, str):
            break

        # Otherwise, update the progress bar and move on to the next candlestick
        else:
            progress_bar.update()

    # Output the performance
    performance: IBacktestPerformance = position.get_performance()

    # Finally, return the results
    return _build_evaluation_result(
        early_stopping=early_stopping,
        neutral_predictions=neutral_predictions,
        increase=increase,
        increase_successful=increase_successful,
        decrease=decrease,
        decrease_successful=decrease_successful,
        increase_outcomes=increase_outcomes,
        decrease_outcomes=decrease_outcomes,
        performance=performance
    )









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
            return pred["md"][0]["up"]
        else:
            return pred["md"][0]["dp"]

    # Otherwise, stop the execution
    else:
        raise ValueError(f"Cannot extract the metadata value from an invalid model type {model_type}")






def _perform_early_stopping_evaluation(
    current_points: float,
    eval_checkpoint_1: bool,
    eval_checkpoint_2: bool,
    eval_checkpoint_3: bool,
    eval_checkpoint_4: bool,
    longs_num: int,
    shorts_num: int
) -> Union[str, None]:
    """Verifies if the model evaluation should be stopped early. If so, it returns
    a string describing the reason. Otherwise, returns None. The Model Evaluation will stop early if:
    1) The model reaches -20 points
    2) The model has less than 1 long or 1 short at the first early stopping checkpoint (15% of the dataset)
    3) The model has less than 3 longs or 3 shorts at the first early stopping checkpoint (30% of the dataset)
    4) The model has less than 10 longs or 10 shorts at the second early stopping checkpoint (50% of the dataset)
    5) The model has less than 15 longs or 15 shorts at the third early stopping checkpoint (70% of the dataset)

    Args:
        current_points: float
            The points that have been collected by the model so far.
        eval_checkpoint_1: bool
        eval_checkpoint_2: bool
        eval_checkpoint_3: bool
        eval_checkpoint_4: bool
            The checkpoint evaluation that should be performed based on the progress.
        longs_num: int
        shorts_num: int
            The number of long and short positions predicted by the model so far.
    """
    # Make sure the min points has not been reached
    if current_points <= -20:
        return "The model evaluation was stopped because the model reached -20 points."

    # If the first checkpoint should be evaluated, make sure it has the min required positions
    elif eval_checkpoint_1 and (longs_num < 1 or shorts_num < 1):
        return "The model evaluation was stopped because the model had less than 1 long or short during the first checkpoint."

    # If the second checkpoint should be evaluated, make sure it has the min required positions
    elif eval_checkpoint_2 and (longs_num < 3 or shorts_num < 3):
        return "The model evaluation was stopped because the model had less than 3 longs or shorts during the second checkpoint."

    # If the third checkpoint should be evaluated, make sure it has the min required positions
    elif eval_checkpoint_3 and (longs_num < 10 or shorts_num < 10):
        return "The model evaluation was stopped because the model had less than 10 longs or shorts during the third checkpoint."

    # If the third checkpoint should be evaluated, make sure it has the min required positions
    elif eval_checkpoint_4 and (longs_num < 15 or shorts_num < 15):
        return "The model evaluation was stopped because the model had less than 15 longs or shorts during the fourth checkpoint."







def _build_evaluation_result(
    early_stopping: Union[str, None],
    neutral_predictions: int,
    increase: List[float], 
    increase_successful: List[float], 
    decrease: List[float], 
    decrease_successful: List[float], 
    increase_outcomes: int,
    decrease_outcomes: int,
    performance: IBacktestPerformance, 
) -> IModelEvaluation:
    """Outputs the model's evaluation result based on all the collected data.

    Args:
        early_stopping: Union[str, None]
            The reason why the evaluation was stopped (if applies)
        neutral_predictions: int
            The number of neutral predictions generated by the model.
        increase: List[float]
            The list of increase predictions' payloads.
        increase_successful: List[float]
            The list of successful increase predictions' payloads.
        decrease: List[float]
            The list of decrease predictions' payloads.
        decrease_successful: List[float]
            The list of deccessful increase predictions' payloads.
        increase_outcomes: int
            The number of real increase outcomes.
        decrease_outcomes: int
            The number of real decrease outcomes.
        performance: IBacktestPerformance
            Performance details provided by the Position Class

    Returns:
        IModelEvaluation
    """
    # Initialize the position type lengths
    increase_successful_num: int = len(increase_successful)
    decrease_successful_num: int = len(decrease_successful)

    # Finally, return the results
    return {
        # Early Stopping
        "early_stopping": early_stopping,

        # Neutral Predictions
        "neutral_predictions": neutral_predictions,

        # Positions
        "positions": performance["positions"],

        # Points Median
        "points_median": median(performance["points_hist"]),

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