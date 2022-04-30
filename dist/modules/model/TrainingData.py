from os import makedirs
from os.path import exists
from typing import List, Dict, Tuple, Union
from json import dumps
from pandas import DataFrame, Series
from tqdm import tqdm
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.model import ITrainingDataConfig, SingleModel, ITrainingDataActivePosition, IPrediction, \
    ITrainingDataReceipt, ITrainingDataPredictionAnalysis




class TrainingData:
    """TrainingData Class

    Generates the data to be used to train Decision Models based on the provided configuration.

    Class Properties:
        OUTPUT_PATH: str
            The directory in which the generated training data will be places.

    Instance Properties:
        id: str
            The identification of the training data. This value is generated based on the single models
            that will be used to generate the training data.
        start: int
            The open timestamp of the first 1m candlestick.
        end: int
            The close timestamp of the last 1m candlestick.
        up_percent_change: float
            The percentage that needs to go up to close an up position
        down_percent_change: float
            The percentage that needs to go down to close a down position
        single_models: List[SingleModel]
            The list of single models that will be used to generate the training data.
        df: DataFrame
            Pandas Dataframe containing all the features and labels populated every time a position
            is closed. This DF will be dumped as a csv once the process completes.
        active: Union[ITrainingDataActivePosition, None]
            Dictionary containing all the single model predictions as well as the up and down price
            details.
    """

    # Directory where results will be dumped
    OUTPUT_PATH: str = './training_data'



    ## Init ##

    def __init__(self, config: ITrainingDataConfig):
        """Initializes the Training Data Instance as well as the candlesticks.

        Args:
            config: ITrainingDataConfig
                The configuration that will be used to generate the training data.

        Raises:
            ValueError:
                If less than 5 single models are provided.
                If the single models don't have the same lookback.
        """
        # Make sure that at least 5 single models were provided
        if len(config['single_models']) < 5:
            raise ValueError(f"A minimum of 5 single models are required in order to generate the training data. \
                Received: {len(config['single_models'])}")

        # Initialize the models to be tested
        self.single_models: List[SingleModel] = [SingleModel(m) for m in config['single_models']]
        
        # Make sure all the single models have the same lookback
        if not all(m.lookback == self.single_models[0].lookback for m in self.single_models):
            raise ValueError("All single models must have the exact same lookback in order to generate the training data.")

        # Generate the training data ID
        self.id: str = ''.join([m.id for m in self.single_models])

        # Initialize the candlesticks
        Candlestick.init(self.single_models[0].lookback, config.get('start'), config.get('end'))

        # Init the start and end
        self.start: int = int(Candlestick.DF.iloc[0]['ot'])
        self.end: int = int(Candlestick.DF.iloc[-1]['ct'])

        # Postitions Up & Down percent change requirements
        self.up_percent_change: float = config['up_percent_change']
        self.down_percent_change: float = config['down_percent_change']

        # Initialize the DF
        self.df: DataFrame = DataFrame(data={column_name: [] for column_name in [m.id for m in self.single_models] + ['up', 'down']})

        # Initialize the active position
        self.active: Union[ITrainingDataActivePosition, None] = None





    ## Execution ##




    def run(self) -> None:
        """Runs the Training Data Process and stores the results once it completes.
        """
        # Init the progress bar
        progress_bar = tqdm( bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=round(Candlestick.DF.shape[0], 0))

        # Init the time in which the execution started
        execution_start: int = Utils.get_time()

        # Iterate over each 1 minute candlestick
        for candlestick_index, candlestick in Candlestick.DF.iterrows():
            # Check if it is the last candlestick
            is_last_candlestick: bool = candlestick_index == Candlestick.DF.index[-1]

            # Active Position
            # If there is an active position, check it against the new candlestick.
            if self.active != None:
                self._check_position(candlestick)

            # Inactive Position
            # If there is not an active position and it is not the last candlestick it opens a new position.
            elif (self.active == None) and (not is_last_candlestick):
                self._open_position(candlestick)

            # Update the progress bar
            progress_bar.update()

        # Save the results
        self._save_results(execution_start)







    ## Positions Management ##



    def _open_position(self, candlestick: Series) -> None:
        """Creates a new position based on the open_price and the single models.

        Args:
            candlestick: Series
                The current candlestick.
        """
        # Calculate the position range
        up_price, down_price = self._get_position_range(candlestick['o'])

        # Generate the single model predictions
        preds: Dict = {m.id: self._get_prediction_result(m.predict(candlestick['ot'], enable_cache=True)) for m in self.single_models}

        # Finally, populate the active position dict
        self.active = {
            'up_price': up_price,
            'down_price': down_price,
            'row': preds
        }





    def _get_position_range(self, open_price: float) -> Tuple[float, float]:
        """Calculates the prices that will be used to determine if the 
        price went up or down.

        Args:
            open_price: float
                The current candlestick's open price.
        
        Returns:
            Tuple[float, float] (up_price, down_price)
        """
        return Utils.alter_number_by_percentage(open_price, self.up_percent_change), \
            Utils.alter_number_by_percentage(open_price, -(self.down_percent_change))





    def _get_prediction_result(self, pred: IPrediction) -> int:
        """Given a Single Model Prediction, it will convert its value to 
        the value required by the decision model to learn properly.
        Long (1) = 2, Short (-1) = 1, Neutral (0) = 0.

        Args:
            pred: IPrediction
                A prediction generated by a Single Model.

        Returns:
            int (0|1|2)
        """
        if pred['r'] == 1:
            return 2
        elif pred['r'] == -1:
            return 1
        else:
            return 0





    def _check_position(self, candlestick: Series) -> None:
        """Checks the state of the active position based on the current 
        candlestick. If the high touched the up_price, the position is closed
        as 'up'. On the other hand, if the low touched the down_price, the position
        is closed as 'down'.

        Args:
            candlestick: Series
                The current candlestick.

        Returns:
            None
        """
        # Check if the high touched the up_price
        if candlestick['h'] >= self.active['up_price']:
            self._close_position(True)
        
        # Check if the low touched the down_price
        elif candlestick['l'] <= self.active['down_price']:
            self._close_position(False)






    def _close_position(self, up: bool) -> None:
        """Completes the row in the active position, appends it to the DataFrame and 
        finally sets the local active position as None.

        Args:
            up: bool
                If the price went up or not.
        """
        # Complete the row in the active position
        self.active['row']['up'] = 1 if up else 0
        self.active['row']['down'] = 1 if not up else 0

        # Append the row to the DataFrame
        self.df.append([self.active['row']])

        # Unset the active position
        self.active = None












    ## Results ## 




    def _save_results(self, execution_start: int) -> None:
        """Creates all the required directories and dumps the CSV and the Receipt File.

        Args:
            execution_start: int
                The time in which the execution started.
        """
        # Init the current time
        current_time: int = Utils.get_time()

        # If the output directory doesn't exist, create it
        if not exists(TrainingData.OUTPUT_PATH):
            makedirs(TrainingData.OUTPUT_PATH)

        # Create the results directory
        result_dir: str = f"{TrainingData.OUTPUT_PATH}/{current_time}"
        makedirs(result_dir)

        # Create the CSV File
        self.df.to_csv(f"{result_dir}/data.csv", index=False)

        # Write the Receipt File
        with open(f"{result_dir}/receipt.json", "w") as receipt_file:
            receipt_file.write(dumps(self._get_receipt(execution_start, current_time), indent=4))





    def _get_receipt(self, execution_start: int, current_time: int) -> ITrainingDataReceipt:
        """Builds a receipt based on all the data collected during the execution.

        Args:
            execution_start: int
                The time in which the execution started.
            current_time: int
                The current time (used to create the output directory).

        Returns:
            ITrainingDataReceipt
        """
        return {
            'id': self.id,
            'creation': current_time,
            'start': self.start,
            'end': self.end,
            'up_percent_change': self.up_percent_change,
            'down_percent_change': self.down_percent_change,
            'single_models': [m.get_model() for m in self.single_models],
            'duration_minutes': Utils.from_milliseconds_to_minutes(current_time - execution_start),
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'price_action_analysis': {
                'up': Utils.get_percentage_out_of_total(self.df['up'].value_counts()[1], self.df.shape[0]),
                'down': Utils.get_percentage_out_of_total(self.df['down'].value_counts()[1], self.df.shape[0]),
            },
            'predictions_analysis': {m.id: self._get_prediction_analysis_for_model(m.id) for m in self.single_models}
        }


    



    def _get_prediction_analysis_for_model(self, model_id: str) -> ITrainingDataPredictionAnalysis:
        """Retrieves the prediction analysis for a given model.

        Args:
            model_id: str
                The ID of the model.
        
        Return:
            ITrainingDataPredictionAnalysis
        """
        return {
            'long': Utils.get_percentage_out_of_total(self.df[model_id].value_counts()[2], self.df.shape[0]),
            'short': Utils.get_percentage_out_of_total(self.df[model_id].value_counts()[1], self.df.shape[0]),
            'neutral': Utils.get_percentage_out_of_total(self.df[model_id].value_counts()[0], self.df.shape[0]),
        }