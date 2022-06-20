from typing import List, Dict
from os import makedirs, listdir, remove
from os.path import exists, isfile, join
from json import load, dumps
from numpy import median, mean, arange
from modules.types import IBacktestResult, IModelResult, ICombinationResult, IRegressionSelectionFile
from modules.utils.Utils import Utils
from modules.backtest.BacktestPath import BACKTEST_PATH




class RegressionSelection:
    """RegressionSelection Class

    This class takes any number of Backtest Results, organizes them by TakeProfit/StopLoss
    combinations and selects the top performing models based on their points median. Once the 
    process completes evaluating models, it will also evaluate TakeProfit/StopLoss combinations
    by the mean of the selected models' points median.

    Class Properties:
        ...

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
        backtest_result_file_names: List[str]
            The list of backtest result files that were extracted. These files will be deleted on completion if
            clean_results_dir is enabled.
    """



    ## Initialization ##

    def __init__(self, models_limit: int, clean_results_dir: bool):
        """Initializes the RegressionSelection Instance and prepares it
        to be executed.

        Args:
            models_limit: int
                The number of models that will be selected per combination.
            clean_results_dir: bool
                If enabled, it will delete all the backtest files from the results directory on 
                completion.
        
        Raises:
            ValueError:
                If the models_limit is smaller than 5.
        """
        # Make sure that the limit is at least 5
        if models_limit < 5:
            raise ValueError(f"The limit of models should be set to at least 5. Received: {models_limit}")

        # Initialize the config args
        self.models_limit: int = models_limit
        self.clean_results_dir: bool = clean_results_dir

        # Generate the ID
        self.id: str = Utils.generate_uuid4()

        # Initialize the date range to be populated once the backtest files are loaded
        self.start: int = 0
        self.end: int = 0

        # Initialize the file names to be deleted at the end of the process (if enabled)
        self.backtest_result_file_names: List[str] = []


        






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
        print(f"{self.id}:")
        # Extract all the backtest results
        print("    1/4) Extracting Backtest Results...")
        backtests: List[IBacktestResult] = self._get_backtest_results()

        # Build the Regression Selection
        print("    2/4) Building Selection...")
        selection: IRegressionSelectionFile = self._build_regression_selection(backtests)

        # Save the Regression Selection
        print("    3/4) Saving Selection...")
        self._save_regression_selection(selection)

        # Clean Backtest Result Files if enabled
        if self.clean_results_dir:
            print("    4/4) Deleting Backtest Results...")
            self._delete_backtest_results()
        else:
            print("    4/4) Deleting Backtest Results Skipped")












    ## Backtest Results Extraction ##





    def _get_backtest_results(self) -> List[IBacktestResult]:
        """Extracts the backtests from the results directory and returns general data
        about them. It is also important to note that this method initializes the following
        args:
            start, end & backtest_result_file_names

        Returns:
            List[IBacktestResult]

        Raises:
            RuntimeError:
                If there are no backtest results in the directory.
                If a file is not a valid Backtest Result.
                If the backtest results don't share the same start and end.
        """
        # If the backtest results dir doesnt exist, raise an error and provide instructions
        if not exists(BACKTEST_PATH["results"]):
            makedirs(BACKTEST_PATH["results"])
            raise RuntimeError(f"The {BACKTEST_PATH['results']} directory must exist and contain backtest result files.")

        # Build the list of file names, also make sure at least 1 file has been extracted
        self.backtest_result_file_names = list(
            filter(
                lambda f: ".json" in f, [f for f in listdir(BACKTEST_PATH['results']) if isfile(join(BACKTEST_PATH['results'], f))]
            )
        )
        if len(self.backtest_result_file_names) < 1:
            raise RuntimeError("There must be at least 1 backtest file in the results directory.")

        # Extract the files while validating their integrity
        result_files: List[IBacktestResult] = []
        for name in self.backtest_result_file_names:
            # Load the file which contains a list of backtests
            files: List[IBacktestResult] = load(open(f"{BACKTEST_PATH['results']}/{name}"))

            # Iterate over each file and validate its data and append it to the final list
            for file in files:
                # Make sure the file is valid
                self._is_backtest_result(name, file)

                # Populate the date range in case it hasn't been
                if self.start == 0:
                    self.start = file["backtest"]["start"]
                    self.end = file["backtest"]["end"]

                # The date ranges must be identical
                if file["backtest"]["start"] != self.start or file["backtest"]["end"] != self.end:
                    raise RuntimeError(f"The file {name} date range is different to the previous files.")

                # Append the file to the list
                result_files.append(file)

        # Finally, return the list of extracted files
        return result_files



        



    def _is_backtest_result(self, file_name: str, file: IBacktestResult) -> None:
        """Checks if an extracted dict is a Backtest Result.

        Args:
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
                        raise RuntimeError(f"The file {file_name} is not a valid Backtest Result.")










    ## Regression Selection ## 





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








    def _build_model_results(self, backtests: List[IBacktestResult]) -> Dict[str, List[IModelResult]]:
        """Builds a dict containing lists of model results based on their combination id.
        Note that the model results will be filtered and ordered by median based on the 
        models_limit provided.

        Args:
            backtests: List[IBacktestResult]
                The list of backtests that will be used to build the models' results.

        Returns:
            Dict[str: List[IModelResult]]
        """
        # Init the model results
        model_results: Dict[str, List[IModelResult]] = {}

        # Iterate over each backtest
        for backtest in backtests:
            # Init the combination ID
            id: str = self._get_combination_id(backtest["backtest"]["take_profit"], backtest["backtest"]["stop_loss"])

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
            "points_median": median(backtest["performance"]["points_hist"]),
            "points_median_hist": median_hist,
            "long_num": backtest["performance"]["long_num"],
            "short_num": backtest["performance"]["short_num"],
            "long_acc": backtest["performance"]["long_acc"],
            "short_acc": backtest["performance"]["short_acc"],
            "general_acc": backtest["performance"]["general_acc"],
        }










    def _build_combination_results(self, model_results: Dict[str, List[IModelResult]]) -> List[ICombinationResult]:
        """Builds the combination results based on the model results. The final list is
        sorted by the mean of the model results medians by combination.

        Args:
            model_results: Dict[str, List[IModelResult]]
        
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








    def _get_combination_id(self, take_profit: float, stop_loss: float) -> str:
        """Builds a combination ID based on the take profit and stop loss combination.

        Args:
            take_profit: float
            float, stop_loss: float
        
        Returns:
            str
        """
        return f"TP{str(take_profit)}-SL{str(stop_loss)}"











    ## File Helpers ## 





    def _save_regression_selection(self, selection: IRegressionSelectionFile) -> None:
        """Saves a given selection in the output path based on its id.

        Args:
            selection: IRegressionSelectionFile
                The selection to be stored.
        """
        # If the results directory doesn't exist, create it
        if not exists(BACKTEST_PATH["regression_selections"]):
            makedirs(BACKTEST_PATH["regression_selections"])

        # Write the results on a JSON File
        with open(f"{BACKTEST_PATH['regression_selections']}/{self.id}.json", "w") as outfile:
            outfile.write(dumps(selection))





    


    def _delete_backtest_results(self) -> None:
        """Deletes all the backtest result files in the results directory.
        """
        for fn in self.backtest_result_file_names:
            remove(f"{BACKTEST_PATH['results']}/{fn}")