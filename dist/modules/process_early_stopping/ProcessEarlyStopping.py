from typing import Union, List
from modules._types import IEarlyStoppingProcessName, IProcessEarlyStoppingCheckpointConfig, IProcessEarlyStoppingCheckpoint






# Class
class ProcessEarlyStopping:
    """ProcessEarlyStopping Class

    This class keeps track of a model's progress during any process that is taking place
    and can determine if the process should continue or stop.

    Instance Properties:
        process_name: IEarlyStoppingProcessName
            The name of the process.
        checkpoints: List[IProcessEarlyStoppingCheckpoint]
            The checkpoints to be evaluated accordingly.
        min_points: Union[float, None]
            The minimum number of points a model can have. This check will be skipped if 
            the min_points are not provided.
    """


    def __init__(
        self, 
        process_name: IEarlyStoppingProcessName, 
        candlesticks_num: int, 
        checkpoints: List[IProcessEarlyStoppingCheckpointConfig],
        min_points: Union[float, None] = None
    ):
        """Initializes the EarlyStopping instance based on the provided
        number of candlesticks.

        Args:
            process_name: IEarlyStoppingProcessName
                The name of the process
            candlesticks_num: int
                The number of 1 minute candlesticks that will be used in the 
                process.
            checkpoints: List[IProcessEarlyStoppingCheckpointConfig]
                The list of checkpoint configurations.
            min_points: Union[float, None]
                The minimum number of points allowed.
        """
        # Init the name of the process
        self.process_name: IEarlyStoppingProcessName = process_name

        # Init the checkpoints
        self.checkpoints: List[IProcessEarlyStoppingCheckpoint] = self._build_checkpoints(candlesticks_num, checkpoints)

        # Init the min points allowed
        self.min_points: Union[float, None] = min_points





    
    def _build_checkpoints(
        self, 
        candlesticks_num: int, 
        configs: List[IProcessEarlyStoppingCheckpointConfig]
    ) -> List[IProcessEarlyStoppingCheckpoint]:
        """Builds the checkpoints that will be evaluated based on the provided
        configurations.
        """
        # Init the checkpoints
        checkpoints: List[IProcessEarlyStoppingCheckpoint] = []

        # Iterate over each config
        for config in configs:
            checkpoints.append({
                "index": int(candlesticks_num * config["dataset_percent"]),
                "passed": False,
                "required_longs": config["required_longs"],
                "required_shorts": config["required_shorts"],
                "motive": self._get_checkpoint_stop_motive(config["required_longs"], config["required_shorts"], config["dataset_percent"])
            })


        # Finally, return the checkpoints
        return checkpoints






    def _get_checkpoint_stop_motive(self, required_longs: int, required_shorts: int, dataset_percent: float) -> str:
        """Retrieves the checkpoint motive for which the process should be stopped.

        Args:
            required_longs: int
            required_shorts: int
            dataset_percent: float
        
        Returns: 
            str
        """
        return (
            f"{self.process_name} stopped early because the model had less than "
            f"{required_longs} longs or {required_shorts} shorts at "
            f"{dataset_percent*100}% of the dataset."
        )









    def check(
        self, 
        current_index: int, 
        longs_num: int, 
        shorts_num: int,
        points: Union[float, None] = None,
    ) -> Union[str, None]:
        """Checks if a ModelEvaluation should continue. If so, it returns None.
        If the evaluation should be stopped, it will return a string with the motive.

        Args:
            points: float
                The points accumulated by the model so far.
            current_index: int
                The index of the current candlestick.
            longs_num: int
            shorts_num: int
                The number of longs and shorts the model has predicted so far.
            points: Union[float, None]
                The points collected by the model so far. If none, this evaluation 
                will be skipped

        Returns:
            Union[str, None]
        """
        # Firstly, make sure the points are acceptable if applies
        if isinstance(self.min_points, (int, float)) and points <= self.min_points:
            return f"{self.process_name} stopped early because the model has less than {self.min_points} points."

        # Iterate over each checkpoint
        for i, checkpoint in enumerate(self.checkpoints):
            # Check if the checkpoint should be evaluated
            if not self.checkpoints[i]["passed"] and current_index >= self.checkpoints[i]["index"]:
                # Evaluate the checkpoint and store the result
                self.checkpoints[i]["passed"] = self._checkpoint_passed(checkpoint, longs_num, shorts_num)

                # If the checkpoint did not pass, return the motive
                if not self.checkpoints[i]["passed"]:
                    return self.checkpoints[i]["motive"]

        # If nothing has been returned means the model is doing ok
        return None

    







    def _checkpoint_passed(self, checkpoint: IProcessEarlyStoppingCheckpoint, longs_num: int, shorts_num: int) -> bool:
        """Given a checkpoint, it will verify the model failed or can keep going.

        Args:
            checkpoint: IProcessEarlyStoppingCheckpoint
                The checkpoint to be evaluated.
            longs_num: int 
            shorts_num: int
                The number of positions the model has engaged so far.

        Returns:
            bool
        """
        return longs_num >= checkpoint["required_longs"] and shorts_num >= checkpoint["required_shorts"]