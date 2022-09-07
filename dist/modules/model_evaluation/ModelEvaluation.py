from typing import Union, List
from pandas import DataFrame
from numpy import mean
from tqdm import tqdm
from modules._types import IPrediction, IPositionPerformance, IModelEvaluation, IModelType
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.model.ModelFactory import ModelInstance
from modules.position.Position import Position
from modules.process_early_stopping.ProcessEarlyStopping import ProcessEarlyStopping






def evaluate(
    model: ModelInstance, 
    price_change_requirement: float, 
    progress_bar_description: str,
    discovery_completed: bool
) -> IModelEvaluation:
    """Performs an evaluation on a recently trained model. The evaluation works 
    similarly to the backtests and it is only performed on the test dataset to 
    ensure the model has not yet seen it.

    Args:
        model: ModelInstance
            The instance of the model to be evaluated
        price_change_requirement: float
            Percentage the price needs to increase or decrease in order for a position to be 
            closed.
        progress_bar_description: str
            The description that will be placed in the progress bar.
        discovery_completed: bool
            If the discovery didn't complete, it will not run the evaluation and instead
            return blank results.

    Returns:
        IModelEvaluation
    """
    # Initialize the type of model
    model_type: IModelType = type(model).__name__

    # Init the 1m candlesticks dataframe
    df: DataFrame = _get_candlesticks_df(model.get_lookback())

    # Init evaluation data
    neutral_predictions: int = 0
    increase: List[float] = []
    increase_successful: List[float] = []
    decrease: List[float] = []
    decrease_successful: List[float] = []

    # Init the progress bar
    progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=df.shape[0])
    progress_bar.set_description(progress_bar_description)

    # Init the Position Instance
    position: Position = Position(take_profit=price_change_requirement, stop_loss=price_change_requirement)

    # Idle Until
    # The model will remain in an idle state until a candlestick's ot is greater than this value.
    idle_until: int = 0

    # Last neutral prediction range close time.
    last_neutral_ct: int = 0

    # Early Stopping
    # Init the values that will help evaluate if the model's evaluation should continue 
    # or be aborted in order to save time and resources.
    es: ProcessEarlyStopping = _get_early_stopping(df.shape[0])
    es_motive: Union[str, None] = None

    # Run the evaluation as long as the discovery completed successfully
    if discovery_completed:
        for candlestick_index, candlestick in df.iterrows():
            # Check if it is the last candlestick
            is_last_candlestick: bool = candlestick_index == df.index[-1]

            # Active Position
            # Check the active position against the new candlestick. If the position is closed,
            # process the metrics
            if position.active != None:
                # Check the position against the new candlestick
                closed_position: bool = position.check_position(candlestick)

                # If the position has been closed, process the metrics and enable idle state
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

                    # Handle a decrease prediction
                    else:
                        # Append the decrease prediction to the list
                        decrease.append(metadata_value)
                        
                        # Check if the prediction was correct
                        if position.positions[-1]["o"]:
                            decrease_successful.append(metadata_value)

                    # Enable the idle state
                    idle_until = Utils.add_minutes(candlestick["ct"], Epoch.IDLE_MINUTES_ON_POSITION_CLOSE)

            # Inactive Position
            # A new prediction will be generated if the following is met:
            # 1) There isn't an active position
            # 1) The model isnt idle 
            # 2) It isn't the last candlestick
            # 3) The current prediction range's close time is greater than the last one
            elif (position.active == None) and (candlestick["ot"] > idle_until) and (not is_last_candlestick):
                # Retrieve the current prediction range's close time
                _, last_ct = Candlestick.get_lookback_prediction_range(Epoch.REGRESSION_LOOKBACK, candlestick["ot"])

                # Only predict in new ranges
                if last_ct > last_neutral_ct:
                    # Generate a prediction
                    pred: IPrediction = model.predict(candlestick["ot"])

                    # If the result isn't neutral, open a position
                    if pred["r"] != 0:
                        position.open_position(candlestick, pred)

                    # Otherwise, handle the neutrality
                    else:
                        # Increase the neutral counter
                        neutral_predictions += 1

                        # Set the last_ct in the current range
                        last_neutral_ct = last_ct

            # Perform the early stopping evaluation
            es_motive = es.check(candlestick_index, position.long_num, position.short_num, position.points[-1])

            # Check if the early stopping has been triggered
            if isinstance(es_motive, str):
                break

            # Otherwise, update the progress bar and move on to the next candlestick
            else:
                progress_bar.update()

    # Finally, return the results
    return _build_evaluation_result(
        early_stopping=es_motive,
        increase=increase,
        increase_successful=increase_successful,
        decrease=decrease,
        decrease_successful=decrease_successful,
        performance=position.get_performance(neutral_num=neutral_predictions)
    )






def _get_candlesticks_df(lookback: int) -> DataFrame:
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
    real_start: int = Candlestick.PREDICTION_DF[Candlestick.PREDICTION_DF["ot"] >= Epoch.TRAINING_EVALUATION_START].iloc[lookback]["ot"]

    # Init 1m candlesticks dataframe
    df: DataFrame = Candlestick.DF[Candlestick.DF["ot"] >= real_start]
    df.reset_index(drop=True, inplace=True)

    # Finally, return the 1m candlesticks df
    return df








def _get_early_stopping(candlesticks_num: int) -> ProcessEarlyStopping:
    """Retrieves the Early Stopping Instance to be used on the evaluation.

    Args:
        candlesticks_num: int
            The number of 1 minute candlesticks in the dataset.

    Returns:
        ProcessEarlyStopping
    """
    return ProcessEarlyStopping(
        process_name="ModelEvaluation",
        candlesticks_num=candlesticks_num,
        checkpoints=[
            { "required_longs": 1, "required_shorts": 1, "dataset_percent": 0.15 },
            { "required_longs": 3, "required_shorts": 3, "dataset_percent": 0.3 },
            { "required_longs": 7, "required_shorts": 7, "dataset_percent": 0.5 },
            { "required_longs": 10, "required_shorts": 10, "dataset_percent": 0.7 },
        ],
        min_points=-35
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
    if model_type == "KerasRegressionModel" or model_type == "XGBRegressionModel":
        return Utils.get_percentage_change(pred["md"][0]["pl"][0], pred["md"][0]["pl"][-1])

    # Extract the data from a Classification Model
    elif model_type == "KerasClassificationModel" or model_type == "XGBClassificationModel":
        if pred["r"] == 1:
            return pred["md"][0]["up"]
        else:
            return pred["md"][0]["dp"]

    # Otherwise, stop the execution
    else:
        raise ValueError(f"Cannot extract the metadata value from an invalid model type {model_type}")












def _build_evaluation_result(
    early_stopping: Union[str, None],
    increase: List[float], 
    increase_successful: List[float], 
    decrease: List[float], 
    decrease_successful: List[float],
    performance: IPositionPerformance, 
) -> IModelEvaluation:
    """Outputs the model's evaluation result based on all the collected data.

    Args:
        early_stopping: Union[str, None]
            The reason why the evaluation was stopped (if applies)
        increase: List[float]
            The list of increase predictions' payloads.
        increase_successful: List[float]
            The list of successful increase predictions' payloads.
        decrease: List[float]
            The list of decrease predictions' payloads.
        decrease_successful: List[float]
            The list of deccessful increase predictions' payloads.
        performance: IPositionPerformance
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
        "neutral_num": performance["neutral_num"],

        # Positions
        "positions": performance["positions"],

        # Points Median
        "points_median": performance["points_median"],

        # Prediction counts
        "increase_num": performance["long_num"],
        "increase_successful_num": increase_successful_num,
        "decrease_num": performance["short_num"],
        "decrease_successful_num": decrease_successful_num,

        # Accuracy
        "increase_accuracy": performance["long_acc"],
        "decrease_accuracy": performance["short_acc"],
        "accuracy": performance["general_acc"],
        
        # Predictions Overview 
        "increase_list": increase,
        "increase_max": round(max(increase if performance["long_num"] > 0 else [0]), 2),
        "increase_min": round(min(increase if performance["long_num"] > 0 else [0]), 2),
        "increase_mean": round(mean(increase if performance["long_num"] > 0 else [0]), 2),
        "increase_successful_list": increase_successful,
        "increase_successful_max": round(max(increase_successful if increase_successful_num > 0 else [0]), 2),
        "increase_successful_min": round(min(increase_successful if increase_successful_num > 0 else [0]), 2),
        "increase_successful_mean": round(mean(increase_successful if increase_successful_num > 0 else [0]), 2),
        "decrease_list": decrease,
        "decrease_max": round(max(decrease if performance["short_num"] > 0 else [0]), 2),
        "decrease_min": round(min(decrease if performance["short_num"] > 0 else [0]), 2),
        "decrease_mean": round(mean(decrease if performance["short_num"] > 0 else [0]), 2),
        "decrease_successful_list": decrease_successful,
        "decrease_successful_max": round(max(decrease_successful if decrease_successful_num > 0 else [0]), 2),
        "decrease_successful_min": round(min(decrease_successful if decrease_successful_num > 0 else [0]), 2),
        "decrease_successful_mean": round(mean(decrease_successful if decrease_successful_num > 0 else [0]), 2),

        # Outcomes
        "increase_outcomes": performance["long_outcome_num"],
        "decrease_outcomes": performance["short_outcome_num"]
    }