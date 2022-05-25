from typing import List, Dict, Tuple, Union
from os import makedirs
from os.path import exists
from json import dumps, load
from pandas import DataFrame, Series
from tqdm import tqdm
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.model import Model, ArimaModel, RegressionModel, IPrediction, IModel
from modules.keras_models import KERAS_PATH
from modules.classification import ITrainingDataConfig, ITrainingDataActivePosition, \
    ITrainingDataFile, ITrainingDataPriceActionsInsight, ITrainingDataPredictionInsight, \
        compress_training_data, decompress_training_data




class ClassificationTrainingData:
    """ClassificationTrainingData Class

    Generates the data to be used to train Classification Models based on the provided configuration.

    Class Properties:
        ...

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks and will perform predictions
            with cache disabled
        id: str
            Universally Unique Identifier (uuid4)
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
        models: List[Union[ArimaModel, RegressionModel]]
            The list of ArimaModels that will be used to generate the training data.
        df: DataFrame
            Pandas Dataframe containing all the features and labels populated every time a position
            is closed. This DF will be dumped as a csv once the process completes.
        active: Union[ITrainingDataActivePosition, None]
            Dictionary containing all the Arima Model predictions as well as the up and down price
            details.
    """





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
                If less than 5 Models are provided.
                If a duplicate Model ID is found.
        """
        # Make sure that at least 5 Arima Models were provided
        if len(config["models"]) < 5:
            raise ValueError(f"A minimum of 5 ArimaModels|RegressionModels are required in order to generate \
                the classification training data. Received: {len(config['models'])}")

        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the data that will be populated
        self.id: str = Utils.generate_uuid4()
        self.models: List[Union[ArimaModel, RegressionModel]] = []
        df_data: Dict = {}
        ids: List[str] = []
        lookbacks: List[int] = []

        # Iterate over each Arima Model
        for m in config["models"]:
            # Make sure it isn"t a duplicate
            if m["id"] in ids:
                raise ValueError(f"Duplicate Model ID provided: {m['id']}")

            # Add the initialized model to the list
            self.models.append(Model(m))
            
            # Populate helpers
            df_data[m["id"]] = []
            ids.append(m["id"])
            lookbacks.append(self.models[-1].get_lookback())

        # Initialize the candlesticks if not unit testing
        if not self.test_mode:
            Candlestick.init(max(lookbacks), config.get("start"), config.get("end"))

        # Init the start and end
        self.start: int = int(Candlestick.DF.iloc[0]["ot"])
        self.end: int = int(Candlestick.DF.iloc[-1]["ct"])

        # Postitions Up & Down percent change requirements
        self.up_percent_change: float = config["up_percent_change"]
        self.down_percent_change: float = config["down_percent_change"]

        # Initialize the DF
        df_data["up"] = []
        df_data["down"] = []
        self.df: DataFrame = DataFrame(data=df_data)

        # Initialize the active position
        self.active: Union[ITrainingDataActivePosition, None] = None





    ## Execution ##




    def run(self) -> None:
        """Runs the Training Data Process and stores the results once it completes.
        """
        # Init the progress bar
        progress_bar = tqdm( bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=Candlestick.DF.shape[0])

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
        self._save_training_data(execution_start)

        # Validate the integrity of the saved training data
        self._validate_integrity()









    ## Positions Management ##



    def _open_position(self, candlestick: Series) -> None:
        """Creates a new position based on the open_price and the Models Predictions.

        Args:
            candlestick: Series
                The current candlestick.
        """
        # Calculate the position range
        up_price, down_price = self._get_position_range(candlestick["o"])

        # Generate the Model predictions
        preds: Dict[str, int] = {
            m.id: ClassificationTrainingData.get_prediction_result(
                m.predict(
                    candlestick["ot"], 
                    enable_cache=not self.test_mode
                )
            ) for m in self.models}

        # Finally, populate the active position dict
        self.active = {
            "up_price": up_price,
            "down_price": down_price,
            "row": preds
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
        if candlestick["h"] >= self.active["up_price"]:
            self._close_position(True)
        
        # Check if the low touched the down_price
        elif candlestick["l"] <= self.active["down_price"]:
            self._close_position(False)






    def _close_position(self, up: bool) -> None:
        """Completes the row in the active position, appends it to the DataFrame and 
        finally sets the local active position as None.

        Args:
            up: bool
                If the price went up or not.
        """
        # Complete the row in the active position
        self.active["row"]["up"] = 1 if up else 0
        self.active["row"]["down"] = 1 if not up else 0

        # Append the row to the DataFrame
        self.df = self.df.append(self.active["row"], ignore_index=True)

        # Unset the active position
        self.active = None








    @staticmethod
    def get_prediction_result(pred: IPrediction) -> int:
        """Given a Model Prediction, it will convert its value to 
        the value required by the classification model to learn properly.
        Long (1) = 2, Short (-1) = 1, Neutral (0) = 0.

        Args:
            pred: IPrediction
                A prediction generated by a Arima Model.

        Returns:
            int (0|1|2)
        """
        if pred["r"] == 1:
            return 2
        elif pred["r"] == -1:
            return 1
        else:
            return 0












    ## Results ## 




    def _save_training_data(self, execution_start: int) -> None:
        """Creates all the required directories and dumps the CSV and the Receipt File.

        Args:
            execution_start: int
                The time in which the execution started.
        """
        # If the output directory doesn't exist, create it
        if not exists(KERAS_PATH["classification_training_data"]):
            makedirs(KERAS_PATH["classification_training_data"])

        # Write the Training Data File
        with open(f"{KERAS_PATH['classification_training_data']}/{self.id}.json", "w") as training_data_file:
            training_data_file.write(dumps(self._build_file(execution_start)))





    def _build_file(self, execution_start: int) -> ITrainingDataFile:
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
            "id": self.id,
            "description": self.description,
            "creation": current_time,
            "start": self.start,
            "end": self.end,
            "up_percent_change": self.up_percent_change,
            "down_percent_change": self.down_percent_change,
            "models": [m.get_model() for m in self.models],
            "duration_minutes": Utils.from_milliseconds_to_minutes(current_time - execution_start),
            "price_actions_insight": self._get_price_actions_insight(),
            "predictions_insight": {m.id: self._get_prediction_insight_for_model(m.id) for m in self.models},
            "training_data": compress_training_data(self.df)
        }


    




    def _get_price_actions_insight(self) -> ITrainingDataPriceActionsInsight:
        """Retrieves the price action insight.

        Returns:
            ITrainingDataPriceActionsInsight
        """
        # Init the value counts
        up_count: Series = self.df["up"].value_counts()
        down_count: Series = self.df["down"].value_counts()

        return {
            "up": int(up_count[1]) if up_count.get(1) is not None else 0,
            "down": int(down_count[1]) if down_count.get(1) is not None else 0
        }






    def _get_prediction_insight_for_model(self, model_id: str) -> ITrainingDataPredictionInsight:
        """Retrieves the prediction insight for a given model.

        Args:
            model_id: str
                The ID of the model.
        
        Return:
            ITrainingDataPredictionInsight
        """
        # Init the value counts
        counts: Series = self.df[model_id].value_counts()

        # Return the insights
        return {
            "long": int(counts[2]) if counts.get(2) is not None else 0,
            "short": int(counts[1]) if counts.get(1) is not None else 0,
            "neutral": int(counts[0]) if counts.get(0) is not None else 0
        }








    


    ## Training Data Integrity Validation ##





    def _validate_integrity(self) -> None:
        """Makes sure that the data stored in the file is valid.

        Raises:
            ValueError:
                If any of the properties or the decompressed training data is 
                    not identical to the original data.
        """
        # Open the file
        td_file = open(f"{KERAS_PATH['classification_training_data']}/{self.id}.json")
        td: ITrainingDataFile = load(td_file)

        # Validate general values
        if td["id"] != self.id:
            raise ValueError(f"ID Discrepancy: {str(td['id'])} != {str(self.id)}")
        if td["description"] != self.description:
            raise ValueError(f"Description Discrepancy: {str(td['description'])} != {str(self.description)}")
        if not isinstance(td["creation"], int):
            raise ValueError(f"The creation timestamp is invalid: {str(td['creation'])}")
        if not isinstance(td["start"], int):
            raise ValueError(f"The start timestamp is invalid: {str(td['start'])}")
        if not isinstance(td["end"], int):
            raise ValueError(f"The end timestamp is invalid: {str(td['end'])}")
        if not isinstance(td["duration_minutes"], int):
            raise ValueError(f"The duration_minutes is invalid: {str(td['duration_minutes'])}")
        if td["up_percent_change"] != self.up_percent_change:
            raise ValueError(f"Up Percent Change Discrepancy: {str(td['up_percent_change'])} != {str(self.up_percent_change)}")
        if td["down_percent_change"] != self.down_percent_change:
            raise ValueError(f"Down Percent Change Discrepancy: {str(td['down_percent_change'])} != {str(self.down_percent_change)}")

        # Validate the models
        models: List[IModel] = [m.get_model() for m in self.models]
        for i in range(len(models) - 1):
            if td["models"][i] != models[i]:
                print(td['models'][i])
                print(models[i])
                raise ValueError(f"There is a discrepancy in the models configurations.")

        # Validate the insights
        price_actions_insight: ITrainingDataPriceActionsInsight = self._get_price_actions_insight()
        if td["price_actions_insight"] != price_actions_insight:
            raise ValueError(f"Price Action Insight Discrepancy: {str(td['price_actions_insight'])} != {str(price_actions_insight)}")
        predictions_insight: ITrainingDataPredictionInsight = {m.id: self._get_prediction_insight_for_model(m.id) for m in self.models}
        if td["predictions_insight"] != predictions_insight:
            raise ValueError(f"Predictions Insight Discrepancy: {str(td['predictions_insight'])} != {str(predictions_insight)}")

        # Validate the training data
        decompressed: DataFrame = decompress_training_data(td["training_data"])
        if not self.df.equals(decompressed):
            print(self.df.head())
            print(decompressed.head())
            raise ValueError(f"There is a discrepancy in the training data that was loaded and decompressed.")


            



    
