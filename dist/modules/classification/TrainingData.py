from os import makedirs
from os.path import exists
from typing import List, Dict, Tuple, Union
from json import dumps
from pandas import DataFrame, Series
from tqdm import tqdm
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.model import ArimaModel, IPrediction
from modules.classification import ITrainingDataConfig, ITrainingDataActivePosition, \
    ITrainingDataFile, ITrainingDataPriceActionsInsight, ITrainingDataPredictionInsight




class TrainingData:
    """TrainingData Class

    Generates the data to be used to train Decision Models based on the provided configuration.

    Class Properties:
        OUTPUT_PATH: str
            The directory in which the generated training data will be places.

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks and will perform predictions
            with cache disabled
        id: str
            Universally Unique Identifier (uuid4)
        arima_id: str
            A secondary ID generated based on the ArimaModels.
        description: str
            Summary describing the purpose and expectations of the Training Data Generation.
        start: int
            The open timestamp of the first 1m candlestick.
        end: int
            The close timestamp of the last 1m candlestick.
        up_percent_change: float
            The percentage that needs to go up to close an up position
        down_percent_change: float
            The percentage that needs to go down to close a down position
        arima_models: List[ArimaModel]
            The list of ArimaModels that will be used to generate the training data.
        df: DataFrame
            Pandas Dataframe containing all the features and labels populated every time a position
            is closed. This DF will be dumped as a csv once the process completes.
        active: Union[ITrainingDataActivePosition, None]
            Dictionary containing all the Arima Model predictions as well as the up and down price
            details.
    """

    # Directory where results will be dumped
    OUTPUT_PATH: str = './training_data'



    ## Init ##

    def __init__(self, config: ITrainingDataConfig, test_mode: bool = False):
        """Initializes the Training Data Instance as well as the candlesticks.

        Args:
            config: ITrainingDataConfig
                The configuration that will be used to generate the training data.
            test_mode: bool
                Indicates if the execution is running from unit tests.

        Raises:
            ValueError:
                If less than 5 Arima Models are provided.
                If a duplicate Arima Model is found.
                If the Arima Models don't have the same lookback.
        """
        # Make sure that at least 5 Arima Models were provided
        if len(config['arima_models']) < 5:
            raise ValueError(f"A minimum of 5 Arima Models are required in order to generate the training data. \
                Received: {len(config['arima_models'])}")

        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the description
        self.description: str = config['description']

        # Initialize the data that will be populated
        self.id: str = Utils.generate_uuid4()
        self.arima_id: str = ''
        self.arima_models: List[ArimaModel] = []
        df_data: Dict = {}
        first_lookback: Union[int, None] = None

        # Iterate over each Arima Model
        for m in config['arima_models']:
            # Make sure it isn't a duplicate
            if m['id'] in self.arima_id:
                raise ValueError(f"Duplicate Arima Model provided: {m['id']}")

            # Populate Instance Data
            self.arima_id = self.arima_id + m['id']
            self.arima_models.append(ArimaModel(m))
            df_data[m['id']] = []
            
            # Make sure the lookbacks are identical
            if first_lookback == None:
                first_lookback = self.arima_models[0].get_lookback() 
            if self.arima_models[-1].get_lookback() != first_lookback:
                raise ValueError(f"Arima Model lookback missmatch: {self.arima_models[-1].lookback} != {first_lookback}")

        # Initialize the candlesticks if not unit testing
        if not self.test_mode:
            Candlestick.init(self.arima_models[0].lookback, config.get('start'), config.get('end'))

        # Init the start and end
        self.start: int = int(Candlestick.DF.iloc[0]['ot'])
        self.end: int = int(Candlestick.DF.iloc[-1]['ct'])

        # Postitions Up & Down percent change requirements
        self.up_percent_change: float = config['up_percent_change']
        self.down_percent_change: float = config['down_percent_change']

        # Initialize the DF
        df_data['up'] = []
        df_data['down'] = []
        self.df: DataFrame = DataFrame(data=df_data)

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
        """Creates a new position based on the open_price and the Arima Models.

        Args:
            candlestick: Series
                The current candlestick.
        """
        # Calculate the position range
        up_price, down_price = self._get_position_range(candlestick['o'])

        # Generate the Arima Model predictions
        preds: Dict[str, int] = {m.id: self._get_prediction_result(m.predict(candlestick['ot'], enable_cache=not self.test_mode)) for m in self.arima_models}

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
        """Given a Arima Model Prediction, it will convert its value to 
        the value required by the decision model to learn properly.
        Long (1) = 2, Short (-1) = 1, Neutral (0) = 0.

        Args:
            pred: IPrediction
                A prediction generated by a Arima Model.

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
        self.df = self.df.append(self.active['row'], ignore_index=True)

        # Unset the active position
        self.active = None












    ## Results ## 




    def _save_results(self, execution_start: int) -> None:
        """Creates all the required directories and dumps the CSV and the Receipt File.

        Args:
            execution_start: int
                The time in which the execution started.
        """
        # If the output directory doesn't exist, create it
        if not exists(TrainingData.OUTPUT_PATH):
            makedirs(TrainingData.OUTPUT_PATH)

        # Write the Training Data File
        with open(f"{TrainingData.OUTPUT_PATH}/{self.id}.json", "w") as training_data_file:
            training_data_file.write(dumps(self._get_file(execution_start)))





    def _get_file(self, execution_start: int) -> ITrainingDataFile:
        """Builds the Training Data File containing all the data collected.

        Args:
            execution_start: int
                The time in which the execution started.
            current_time: int
                The current time (used to create the output directory).

        Returns:
            ITrainingDataFile
        """
        # Initialize the current time
        current_time: int = Utils.get_time()

        # Convert all the values to integers
        self.df = self.df.astype(int)

        # Return the File Data
        return {
            'id': self.id,
            'arima_id': self.arima_id,
            'description': self.description,
            'creation': current_time,
            'start': self.start,
            'end': self.end,
            'up_percent_change': self.up_percent_change,
            'down_percent_change': self.down_percent_change,
            'arima_models': [m.get_model() for m in self.arima_models],
            'duration_minutes': Utils.from_milliseconds_to_minutes(current_time - execution_start),
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'price_action_insight': self._get_price_action_insight(),
            'predictions_insight': {m.id: self._get_prediction_insight_for_model(m.id) for m in self.arima_models},
            'training_data': self.df.to_dict('records')
        }


    




    def _get_price_action_insight(self) -> ITrainingDataPriceActionsInsight:
        """Retrieves the price action insight.

        Returns:
            ITrainingDataPriceActionsInsight
        """
        return {
            'up': Utils.get_percentage_out_of_total(self.df['up'].value_counts()[1], self.df.shape[0]),
            'down': Utils.get_percentage_out_of_total(self.df['down'].value_counts()[1], self.df.shape[0]),
        }






    def _get_prediction_insight_for_model(self, model_id: str) -> ITrainingDataPredictionInsight:
        """Retrieves the prediction insight for a given model.

        Args:
            model_id: str
                The ID of the model.
        
        Return:
            ITrainingDataPredictionInsight
        """
        return {
            'long': Utils.get_percentage_out_of_total(self.df[model_id].value_counts()[2], self.df.shape[0]),
            'short': Utils.get_percentage_out_of_total(self.df[model_id].value_counts()[1], self.df.shape[0]),
            'neutral': Utils.get_percentage_out_of_total(self.df[model_id].value_counts()[0], self.df.shape[0]),
        }