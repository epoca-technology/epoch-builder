from typing import List, Union
from os import makedirs
from os.path import exists
from json import dumps
from tqdm import tqdm
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.model import IModel, Model, IPrediction, ArimaModel, RegressionModel, ClassificationModel
from modules.backtest import BACKTEST_PATH, IBacktestConfig, Position, IBacktestPerformance, IBacktestResult




class Backtest:
    """Backtest Class

    This class performs backtesting on a batch of models and outputs the results to a json file
    that can be analyzed in the GUI.

    Class Properties:
        ...

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks and will perform predictions
            with cache disabled
        Identification: 
            id: str
                The identification/description of the Backtest Instance. This value must be compatible
                with file systems as it will be part of the result name like {BACKTEST_ID}_{TIMESTAMP}.json
            description: str
                A description to specify the purpose of the backtest.

        Backtest Start and End Range:
            start: int
                The open timestamp of the first 1m candlestick.
            end: int
                The close timestamp of the last 1m candlestick.

        Positions Configuration:
            take_profit: float
                The take profit percentage that will be used in positions.
            stop_loss: float
                The stop loss percentage that will be used in positions.
            idle_minutes_on_position_close: int
                The number of minutes that the model will be idle when a position is closed.

        Models:
            models: List[Union[ArimaModel, RegressionModel, ClassificationModel]]
                The list of models that will be backtested
            results: List[IBacktestResult]
                The list of results by model. This list will be sorted by points prior to outputting it
                to a json file.
    """








    ## Initialization ##


    def __init__(self, config: IBacktestConfig, test_mode: bool = False):
        """Initializes the Backtesting Instance as well as the Candlesticks.

        Args:
            config: IBacktestConfig
                The configuration that will be used during the backtesting process
            test_mode: bool
                Indicates if the execution is running from unit tests.
        """
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # ID & Description
        self.id: str = config['id']
        self.description: str = config['description']

        # Initialize the models to be tested
        self.models: List[Union[ArimaModel, RegressionModel, ClassificationModel]] = [Model(m) for m in config['models']]
        self.results: List[IBacktestResult] = []

        # Initialize the candlesticks based on the models' lookback and the provided start and end dates
        if not self.test_mode:
            Candlestick.init(self.models[0].get_lookback(), config.get('start'), config.get('end'))
        
        # Init the start and end
        self.start: int = int(Candlestick.DF.iloc[0]['ot'])
        self.end: int = int(Candlestick.DF.iloc[-1]['ct'])

        # Postitions Take Profit & Stop Loss
        self.take_profit: float = config['take_profit']
        self.stop_loss: float = config['stop_loss']

        # Idle on position close
        self.idle_minutes_on_position_close: int = config['idle_minutes_on_position_close']










    ## BACKTESTING ##






    def run(self) -> None:
        """Runs the Backtest Instance and stores the results when it is
        finished.
        """
        # Init the progress bar
        progress_bar = tqdm( bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=round(Candlestick.DF.shape[0] * len(self.models), 0))

        # Iterate over each model
        for model_index, model in enumerate(self.models):
            # Set the Model's ID on the progress bar
            progress_bar.set_description(f"{model_index + 1}/{len(self.models)}) {model.id}")

            # Init the time in which the model's backtesting started
            model_start: int = Utils.get_time()

            # Init the Position Instance
            position: Position = Position(self.take_profit, self.stop_loss)

            # Idle Until
            # The model will remain in an idle state until a candlestick's ot is greater than this value.
            idle_until: int = 0

            # Iterate over each 1 minute candlestick
            for candlestick_index, candlestick in Candlestick.DF.iterrows():
                # Check if it is the last candlestick
                is_last_candlestick: bool = candlestick_index == Candlestick.DF.index[-1]

                # Active Position
                # If there is an active position, check it against the new candlestick and
                # enable idling if it was closed no matter the outcome
                if position.active != None:
                    # Check the position with the new candlestick
                    closed_position: bool = position.check_position(candlestick)

                    # Enable idling if a position has been closed
                    if closed_position:
                        idle_until = Utils.add_minutes(candlestick['ct'], self.idle_minutes_on_position_close)

                # Inactive Position
                # If there is not an active position, a prediction will be generated and a position will be opened (if applies).
                # To perform predictions, the following criteria must be met:
                # 1) The model isnt idle 
                # 2) It isn't the last candlestick 
                elif (position.active == None) and (candlestick['ot'] > idle_until) and (not is_last_candlestick):
                    # Perform a prediction
                    pred: IPrediction = model.predict(candlestick['ot'], enable_cache=not self.test_mode)

                    # If the result isn't neutral, open a position
                    if pred['r'] != 0:
                        position.open_position(candlestick, pred)

                # Update the progress bar
                progress_bar.update()

            # Append the Model's Results
            self._append_result(model_start, model.get_model(), position.get_performance())
        
        # Save the results
        self._save_results()














    ## Results ##




    def _append_result(self, model_start: int, model: IModel, performance: IBacktestPerformance) -> None:
        """Appends the model's results to the list once it has completed backtesting.

        Args:
            model_start: int
                The time in which the model's backtesting started.
            model: IModel
                The model that went through the backtesting.
            performance: IBacktestPerformance
                The performance of the model.
        """
        # Init the end time
        model_end: int = Utils.get_time()

        # Apend the model result
        self.results.append({
            'backtest': {
                'id': self.id,
                'description': self.description,
                'start': self.start,
                'end': self.end,
                'take_profit': self.take_profit,
                'stop_loss': self.stop_loss,
                'idle_minutes_on_position_close': self.idle_minutes_on_position_close,
                'model_start': model_start,
                'model_end': model_end,
                'model_duration': Utils.from_milliseconds_to_minutes(model_end - model_start)
            },
            'model': model,
            'performance': performance
        })








    def _save_results(self):
        """Saves the backtest results into the system's directory.
        """
        # If the results directory doesn't exist, create it
        if not exists(BACKTEST_PATH['results']):
            makedirs(BACKTEST_PATH['results'])

        # Write the results on a JSON File
        with open(f"{BACKTEST_PATH['results']}/{self._get_result_file_name()}", "w") as outfile:
            outfile.write(dumps(self.results))









    def _get_result_file_name(self) -> str:
        """Returns the result file name based on the id of the backtest and
        the current time.

        Returns:
            str
        """
        return f"{self.id}_{str(Utils.get_time())}.json"
