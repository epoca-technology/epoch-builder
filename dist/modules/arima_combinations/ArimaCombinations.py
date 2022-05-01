from os import makedirs
from os.path import exists
from typing import List, Tuple, Union
from itertools import product
from json import dumps
from modules.backtest import IBacktestConfig
from modules.model import IModel
from modules.arima_combinations import IArimaCombinationsConfig, IArimaCombination




class ArimaCombinations:
    """ArimaCombinations Class

    This class generates Backtest Configuration Files for all the possible ARIMA(p, d, q) combinations.

    Class Properties:
        OUTPUT_PATH: str
            The path in which the output files will be placed.

    Instance Properties:
        base_id: str
            The base ID that will be use to generate the configuration files.
        description: str
            A description to specify the purpose of the backtest.
        start: Union[int, str, None]
            The date in which the backtest should begin.
        end: Union[int, str, None]
            The date in which the backtest should end.
        take_profit: float
            The take profit percentage that will be used in positions.
        stop_loss: float
            The stop loss percentage that will be used in positions.
        idle_minutes_on_position_close: int
            The number of minutes that the model will be idle when a position is closed.
        focus_number: int
            The number that will be focused when selecting combinations. For example, if the number
            5 is in focus, some possible combinations are: 1,2,5|5,2,1|4,3,5
        batch_size: int
            The maximum number of models that should go in each file.
    """
    
    # Directory where the Backtest configuration files will be placed
    OUTPUT_PATH: str = './backtest_configurations'



    def __init__(self, config: IArimaCombinationsConfig):
        """Initializes the ArimaCombinations instance and prepares it to generate the possible
        combinations.

        Args:
            config: IArimaCombinationsConfig
                The configuration to be used to generate the Backtest Files.
        """
        self.base_id: str = config['id']
        self.description: str = config['description']
        self.start: Union[str, int, None] = config['start']
        self.end: Union[str, int, None] = config['end']
        self.take_profit: float = config['take_profit']
        self.stop_loss: float = config['stop_loss']
        self.idle_minutes_on_position_close: int = config['idle_minutes_on_position_close']
        self.focus_number: int = config['focus_number']
        self.batch_size: int = config['batch_size']





    ## Generator ##


    def generate(self) -> None:
        """Based on the configuration provided, it will generate a series of Backtest Configuration
        Files and place them in the output directory.

        Raises:
            ValueError: 
                If the backtest file already exists.
        """
        # Retrieve the possible combinations
        combinations: List[IArimaCombination] = self._get_combinations()

        # Split the combinations into batches
        batches: List[List[IArimaCombination]] = [combinations[i:i + self.batch_size] for i in range(0, len(combinations), self.batch_size)]
        
        # Save batches individually
        for index, batch in enumerate(batches):
            self._save_batch(batch, index + 1)








    ## Combinations ##





    def _get_combinations(self) -> List[IArimaCombination]:
        """Extracts a lists of valid combinations based on the provided focus_number
        and returns them in a processed format.
        
        Returns:
            List[IArimaCombination]
        """
        # Initialize the range that it will cover
        t = range(self.focus_number + 1)

        # Extract the raw combinations
        raw: List[Tuple[int, int, int]] = list(filter(self._is_combination, set(product(set(t), repeat = 3))))

        # Return the processed combinations
        return list(map(self._process_combination, raw))





    def _is_combination(self, combination: Tuple[int, int, int]) -> bool:
        """Verifies if a provided combination element is a valid combination
        based on the focus_number.

        Args:
            combination: Tuple[int, int, int]
                A possible combination generated by the product helper.
        
        Returns:
            bool
        """
        return self.focus_number in combination and combination[0] != 0 and combination[2] != 0






    def _process_combination(self, combination: Tuple[int, int, int]) -> IArimaCombination:
        """Given a raw combination in Tuple format, it will convert it into a proper dict.

        Args:
            combination: Tuple[int, int, int]
                The raw combination to be processed
        """
        return {'p': combination[0], 'd': combination[1], 'q': combination[2]}








    ## Batch Processing ##



    def _save_batch(self, combinations: List[IArimaCombination], batch_num: int) -> None:
        """Saves a batch of combinations in a Backtest File Format.

        Args:
            combinations: List[IArimaCombination]
                The combinations to be included in the Backtest File.
            batch_num: int
                The number of the batch that is being processed.
        """
        self._save_file({
            "id": f"{self.base_id}_{self.focus_number}_{batch_num}",
            "description": self.description,
            "start": self.start,
            "end": self.end,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "idle_minutes_on_position_close": self.idle_minutes_on_position_close,
            "models": list(map(self._build_model, combinations))
        })






    def _build_model(self, comb: IArimaCombination) -> IModel:
        """Builds a model from a combination.

        Args:
            comb: IArimaCombination
                Combination used to build a model.

        Returns:
            IModel
        """
        return {
            "id": f"A{comb['p']}{comb['d']}{comb['q']}",
            "single_models":[{
                "lookback": 300,
                "arima": { "predictions": 10, "p": comb['p'], "d": comb['d'], "q": comb['q'] },
                "interpreter": { "long": 0.05, "short": 0.05 }
            }]
        }







    def _save_file(self, backtest_file: IBacktestConfig):
        """Saves the backtest file into the output directory.

        Args:
            backtest_file: IBacktestConfig
                The file to be saved in the output directory.

        Raises:
            ValueError: 
                If the backtest file already exists.
        """
        # If the results directory doesn't exist, create it
        if not exists(ArimaCombinations.OUTPUT_PATH):
            makedirs(ArimaCombinations.OUTPUT_PATH)

        # Initialize the backtest directory and the file
        backtest_dir_path: str = f"{ArimaCombinations.OUTPUT_PATH}/{self.base_id}_{self.focus_number}"
        backtest_file_path: str = f"{backtest_dir_path}/{backtest_file['id']}.json"

        # If the backtest directory does not exist, create it
        if not exists(backtest_dir_path):
            makedirs(backtest_dir_path)

        # If the backtest file already exists, raise an error
        if exists(backtest_file_path):
            raise ValueError(f"The file {backtest_file_path} already exists.")

        # Write the results on a JSON File
        with open(f"{backtest_file_path}", "w") as outfile:
            outfile.write(dumps(backtest_file, indent=4))