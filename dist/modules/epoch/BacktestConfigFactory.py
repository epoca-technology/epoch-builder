from typing import List
from modules._types import IBacktestConfig, IModel, IBacktestID
from modules.utils.Utils import Utils
from modules.epoch.EpochFile import EpochFile





class BacktestConfigFactory:
    """BacktestConfigFactory Class

    This singleton manages the creation of Backtest Configuration files for a series of
    processes.

    Class Properties:
        ...
    """








    ## Shortlisted Keras Regressions ##





    ## Shortlisted Keras Classifications ##










    ## Misc Helpers ##





    @staticmethod
    def _save_config(epoch_id: str, config: IBacktestConfig) -> None:
        """Builds a Backtest Configuration File based on provided values.

        Args:
            epoch_id: str
                The ID of the Epoch
            config: IBacktestConfig
                The configuration of the Backtest that will be saved.
        """
        # Build the file's path
        path: str = f"{epoch_id}/{EpochFile.BACKTEST_PATH['configurations']}/{config['id']}.json"

        # Finally, save the file
        Utils.write(path, config, indent=4)







    @staticmethod
    def _build_config(
        backtest_id: IBacktestID,
        description: str,
        take_profit: float,
        stop_loss: float,
        idle_minutes_on_position_close: int,
        models: List[IModel]
    ) -> IBacktestConfig:
        """Builds a Backtest Configuration File based on provided values.

        Args:
            backtest_id: IBacktestID
                The ID of the Backtest.
            description: str
                A brief description of the Backtest.
            take_profit: float
            stop_loss: float
                Position exit combination
            idle_minutes_on_position_close: int
                The number of minutes the models will remain idle on position close.
            models: List[IModel]
                The list of models to be backtested.
        Returns:
            IBacktestConfig
        """
        return {
            "id": backtest_id,
            "description": description,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "idle_minutes_on_position_close": idle_minutes_on_position_close,
            "models": models
        }