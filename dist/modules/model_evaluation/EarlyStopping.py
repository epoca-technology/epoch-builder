from typing import Union
from modules.types import IEarlyStoppingCheckpoint






# Class
class EarlyStopping:
    """EarlyStopping Class

    This class keeps track of a model's progress during an evaluation and determines if it should
    be stopped early.

    Class Properties:
        UNACCEPTABLE_POINTS_MOTIVE: str
        CHECKPOINT_1_MOTIVE: str
        CHECKPOINT_2_MOTIVE: str
        CHECKPOINT_3_MOTIVE: str
        CHECKPOINT_4_MOTIVE: str
            The motives that describe the reason why a Model should not continue the evaluation.

    Instance Properties:
        checkpoint_1: IEarlyStoppingCheckpoint
        checkpoint_2: IEarlyStoppingCheckpoint
        checkpoint_3: IEarlyStoppingCheckpoint
        checkpoint_4: IEarlyStoppingCheckpoint
            The checkpoints in which it will be determined if the model should move forward or not.
    """
    # Early Stopping Moves
    UNACCEPTABLE_POINTS_MOTIVE: str = "The model evaluation was stopped because the model reached -20 points."
    CHECKPOINT_1_MOTIVE: str = "The model evaluation was stopped because the model had less than 1 long or short during the first checkpoint (15% of the dataset)."
    CHECKPOINT_2_MOTIVE: str = "The model evaluation was stopped because the model had less than 3 longs or shorts during the second checkpoint (30% of the dataset)."
    CHECKPOINT_3_MOTIVE: str = "The model evaluation was stopped because the model had less than 7 longs or shorts during the third checkpoint (50% of the dataset)."
    CHECKPOINT_4_MOTIVE: str = "The model evaluation was stopped because the model had less than 10 longs or shorts during the fourth checkpoint (70% of the dataset)."



    def __init__(self, candlesticks_num: int):
        """Initializes the EarlyStopping instance based on the provided
        number of candlesticks.

        Args:
            candlesticks_num: int
                The number of 1 minute candlesticks that will be used in the 
                evaluation.
        """
        # Init the Checkpoint 1
        self.checkpoint_1: IEarlyStoppingCheckpoint = {
            "index": int(candlesticks_num * 0.15),
            "passed": False,
            "required_longs": 1,
            "required_shorts": 1,
            "motive": EarlyStopping.CHECKPOINT_1_MOTIVE,
        }

        # Init the Checkpoint 2
        self.checkpoint_2: IEarlyStoppingCheckpoint = {
            "index": int(candlesticks_num * 0.3),
            "passed": False,
            "required_longs": 3,
            "required_shorts": 3,
            "motive": EarlyStopping.CHECKPOINT_2_MOTIVE,
        }

        # Init the Checkpoint 3
        self.checkpoint_3: IEarlyStoppingCheckpoint = {
            "index": int(candlesticks_num * 0.5),
            "passed": False,
            "required_longs": 7,
            "required_shorts": 7,
            "motive": EarlyStopping.CHECKPOINT_3_MOTIVE,
        }

        # Init the Checkpoint 4
        self.checkpoint_4: IEarlyStoppingCheckpoint = {
            "index": int(candlesticks_num * 0.7),
            "passed": False,
            "required_longs": 10,
            "required_shorts": 10,
            "motive": EarlyStopping.CHECKPOINT_4_MOTIVE,
        }








    def check(
        self, 
        points: float, 
        current_index: int, 
        longs_num: int, 
        shorts_num: int
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

        Returns:
            Union[str, None]
        """
        # Firstly, make sure the points are acceptable
        if points <= -20:
            return EarlyStopping.UNACCEPTABLE_POINTS_MOTIVE

        # Evaluate the first checkpoint if applies
        if not self.checkpoint_1["passed"] and current_index >= self.checkpoint_1["index"]:
            self.checkpoint_1["passed"] = self._checkpoint_passed(self.checkpoint_1, longs_num, shorts_num)
            if not self.checkpoint_1["passed"]:
                return self.checkpoint_1["motive"]

        # Evaluate the second checkpoint if applies
        if not self.checkpoint_2["passed"] and current_index >= self.checkpoint_2["index"]:
            self.checkpoint_2["passed"] = self._checkpoint_passed(self.checkpoint_2, longs_num, shorts_num)
            if not self.checkpoint_2["passed"]:
                return self.checkpoint_2["motive"]

        # Evaluate the third checkpoint if applies
        if not self.checkpoint_3["passed"] and current_index >= self.checkpoint_3["index"]:
            self.checkpoint_3["passed"] = self._checkpoint_passed(self.checkpoint_3, longs_num, shorts_num)
            if not self.checkpoint_3["passed"]:
                return self.checkpoint_3["motive"]

        # Evaluate the fourth checkpoint if applies
        if not self.checkpoint_4["passed"] and current_index >= self.checkpoint_4["index"]:
            self.checkpoint_4["passed"] = self._checkpoint_passed(self.checkpoint_4, longs_num, shorts_num)
            if not self.checkpoint_4["passed"]:
                return self.checkpoint_4["motive"]

        # If nothing has been returned means the model is doing ok
        return None

    







    def _checkpoint_passed(self, checkpoint: IEarlyStoppingCheckpoint, longs_num: int, shorts_num: int) -> bool:
        """Given a checkpoint, it will verify the model failed or can keep going.

        Args:
            checkpoint: IEarlyStoppingCheckpoint
                The checkpoint to be evaluated.
            longs_num: int 
            shorts_num: int
                The number of positions the model has engaged so far.

        Returns:
            bool
        """
        return longs_num >= checkpoint["required_longs"] and shorts_num >= checkpoint["required_shorts"]