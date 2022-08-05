from typing import List, Dict
from numpy import median, mean, arange
from modules.types import IBacktestResult, IModelResult, ICombinationResult, IRegressionSelectionFile,\
    IPositionExitCombinationID, IModelResults, IBacktestID
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.epoch.PositionExitCombination import PositionExitCombination



class RegressionSelection:
    """RegressionSelection Class

    This class takes any number of Backtest Results, organizes them by Position Exit 
    Combinations and selects the top performing models based on their points median. Once the 
    process completes evaluating models, it will also evaluate TakeProfit/StopLoss combinations
    by the mean of the selected models' points median.

    Class Properties:
        BACKTEST_RESULTS: List[IBacktestID]
            The list of backtest result files that will be extracted in order to build the
            Regression Selection.

    Instance Properties
        models_limit: int
            The number of models that will be selected per combination.
        clean_results_dir: bool
            If enabled, it will delete all the backtest files from the results directory on 
            completion.
        id: str
            Universally Unique Identifier (uuid4)
        start: int
            The open timestamp of the first 1m candlestick (Extracted directly from the first backtest result).
        end: int
            The close timestamp of the last 1m candlestick (Extracted directly from the first backtest result).
    """
    BACKTEST_RESULTS: List[IBacktestID] = [
        # Arima Backtests
        "arima_1", "arima_2", "arima_3", "arima_4", "arima_5", "arima_6", "arima_7", "arima_8", "arima_9",

        # Keras Regression Backtests
        "keras_regression",

        # XGBoost Regression Backtests
        #"xgb_regression"
    ]



    ## Initialization ##

    def __init__(self, models_limit: int):
        """Initializes the RegressionSelection Instance and prepares it
        to be executed.

        Args:
            models_limit: int
                The number of models that will be selected per combination.
        
        Raises:
            ValueError:
                If the models_limit is smaller than 5.
        """
        # Make sure that the limit is at least 5
        if models_limit < 5:
            raise ValueError(f"The limit of models should be set to at least 5. Received: {models_limit}")

        # Initialize the config args
        self.models_limit: int = models_limit

        # Generate the ID
        self.id: str = Utils.generate_uuid4()

        # Initialize the date range to be populated once the backtest files are loaded
        self.start: int = 0
        self.end: int = 0


        






    ## Execution ## 





    def run(self) -> None:
        """Executes the RegressionSelection and stores the results once it
        completes.

        Raises:
            RuntimeError:
                If there are no backtest results in the directory.
                If a file is not a valid Backtest Result.
                If the backtest results don't share the same start and end.
        """
        print(f"\n{self.id}:")
        # Extract all the backtest results
        print("    1/3) Extracting Backtest Results...")
        backtests: List[IBacktestResult] = self._get_backtest_results()

        # Build the Regression Selection
        print("    2/3) Building Selection...")
        selection: IRegressionSelectionFile = self._build_regression_selection(backtests)

        # Save the Regression Selection
        print("    3/3) Saving Selection...")
        Epoch.FILE.save_regression_selection(selection)












    ## Backtest Results Extraction ##





    def _get_backtest_results(self) -> List[IBacktestResult]:
        """Extracts the backtests from the results directory and returns general data
        about them. It is also important to note that this method initializes the following
        args:
            self.start, self.end

        Returns:
            List[IBacktestResult]

        Raises:
            RuntimeError:
                If any of the backtest result files is missing.
                If a file is not a valid Backtest Result.
                If the backtest results don't share the same start and end.
        """
        # Init values
        backtest_results: List[IBacktestResult] = []

        # Iterate over each position exit combination
        for pe in PositionExitCombination.get_records():
            # Iterate over each backtest result file
            for backtest_id in RegressionSelection.BACKTEST_RESULTS:
                # Extract the bactest result
                backtest_results: List[IBacktestResult] = \
                    Epoch.FILE.get_backtest_results(backtest_id, pe["take_profit"], pe["stop_loss"])

                # Iterate over each of the results
                for result in backtest_results:
                    # Make sure the file is valid
                    self._is_backtest_result(backtest_id, result)

                    # Populate the date range in case it hasn't been
                    if self.start == 0:
                        self.start = result["backtest"]["start"]
                        self.end = result["backtest"]["end"]

                    # The date ranges must be identical
                    if result["backtest"]["start"] != self.start or result["backtest"]["end"] != self.end:
                        raise RuntimeError(f"The file {backtest_id} date range is different to the previous files.")

                    # Append the result to the list
                    backtest_results.append(result)

        # Finally, return the results
        return backtest_results



        



    def _is_backtest_result(self, backtest_id: IBacktestID, file: IBacktestResult) -> None:
        """Checks if an extracted dict is a Backtest Result.

        Args:
            backtest_id: IBacktestID
                The identifier of the broken backtest result
            file: IBacktestResult
                The file to be checked.

        Raises:
            RuntimeError:
                If a file is not a valid Backtest Result.
        """
        if not isinstance(file, dict) or \
            not isinstance(file["backtest"], dict) or \
                not isinstance(file["model"], dict) or \
                    not isinstance(file["performance"], dict):
                        raise RuntimeError(f"The file {backtest_id} is not a valid Backtest Result.")











    ## Regression Selection Building ## 





    def _build_regression_selection(self, backtests: List[IBacktestResult]) -> IRegressionSelectionFile:
        """Builds a Regression Selection based on a list of Backtest Results.

        Args:
            backtests: List[IBacktestResult]
                The list of backtest results to be analyzed

        Raises:
            RuntimeError:
                If a combination has fewer models than the models_limit provided in the config
        """
        # Build the model results
        model_results: Dict[str, List[IModelResult]] = self._build_model_results(backtests)

        # Build the combination results
        combination_results: List[ICombinationResult] = self._build_combination_results(model_results)

        # Finally, return the regression file
        return {
            "id": self.id,
            "models_limit": self.models_limit,
            "start": self.start,
            "end": self.end,
            "models_num": len(backtests),
            "results": combination_results
        }








    def _build_model_results(self, backtests: List[IBacktestResult]) -> IModelResults:
        """Builds a dict containing lists of model results based on their combination id.
        Note that the model results will be filtered and ordered by median based on the 
        models_limit provided.

        Args:
            backtests: List[IBacktestResult]
                The list of backtests that will be used to build the models' results.

        Returns:
            IModelResults
        """
        # Init the model results
        model_results: IModelResults = {}

        # Iterate over each backtest
        for backtest in backtests:
            # Init the combination ID
            id: IPositionExitCombinationID = PositionExitCombination.get_id(
                take_profit=backtest["backtest"]["take_profit"], 
                stop_loss=backtest["backtest"]["stop_loss"]
            )

            # Build the model result
            result: IModelResult = self._build_model_result(backtest)

            # Add it to the dict safely in case it does not exist yet
            if model_results.get(id) is None:
                model_results[id] = [result]
            else:
                model_results[id].append(result)

        # Sort all the results by median and apply the appropiate slice
        for combination_id in model_results:
            model_results[combination_id] = sorted(model_results[combination_id], key=lambda d: d["points_median"], reverse=True)
            model_results[combination_id] = model_results[combination_id][0:self.models_limit]

        # Finally, return the sorted and filtered model results
        return model_results









    def _build_model_result(self, backtest: IBacktestResult) -> IModelResult:
        """Builds a single model result based on a backtest.

        Args:
            backtest: IBacktestResult

        Returns:
            IModelResult
        """
        # Build the points median history
        median_hist: List[float] = []
        for i in arange(0.1, 1.1, 0.1):
            median_hist.append(median(backtest["performance"]["points_hist"][:int(len(backtest["performance"]["points_hist"]) * i)]))

        # Return the results
        return {
            "model": backtest["model"],
            "points_median": backtest["performance"]["points_median"],
            "points_median_hist": median_hist,
            "long_num": backtest["performance"]["long_num"],
            "short_num": backtest["performance"]["short_num"],
            "long_acc": backtest["performance"]["long_acc"],
            "short_acc": backtest["performance"]["short_acc"],
            "general_acc": backtest["performance"]["general_acc"],
        }










    def _build_combination_results(self, model_results: IModelResults) -> List[ICombinationResult]:
        """Builds the combination results based on the model results. The final list is
        sorted by the mean of the model results medians by combination.

        Args:
            model_results: IModelResults
        
        Returns:
            List[ICombinationResult]
        """
        # Initialize the final list of combination results
        results: List[ICombinationResult] = []

        # Iterate over each combination and add the results to the list
        for combination_id in model_results:
            results.append({
                "combination_id": combination_id,
                "models_num": len(model_results[combination_id]),
                "points_mean": mean([m["points_median"] for m in model_results[combination_id]]),
                "model_results": model_results[combination_id]
            })

        # Finally, sort the list by points mean
        return sorted(results, key=lambda d: d["points_mean"], reverse=True)