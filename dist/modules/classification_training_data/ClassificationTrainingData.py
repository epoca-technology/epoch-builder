from typing import List, Dict, Tuple, Union
from pandas import DataFrame, Series
from math import ceil
from tqdm import tqdm
from modules._types import ITechnicalAnalysis, ITrainingDataConfig, ITrainingDataActivePosition,\
    ITrainingDataFile, ICompressedTrainingData, ITrainingDataSummary
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.model.RegressionModelFactory import RegressionModelFactory, RegressionModel
from modules.technical_analysis.TechnicalAnalysis import TechnicalAnalysis




class ClassificationTrainingData:
    """ClassificationTrainingData Class

    Generates the data to be used to train Classification Models based on the provided configuration.

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks and will perform predictions
            with cache disabled
        regression_selection_id: str
            The ID of the Regression Selection that was used to pick the Regression Models
        id: str
            Universally Unique Identifier (uuid4)
        description: str
            Summary describing the purpose and expectations of the Training Data Generation.
        max_lookback: int
            The maximum lookback among the regressions that will be used to build the data.
        start: int
            The open timestamp of the first 1m candlestick.
        end: int
            The close timestamp of the last 1m candlestick.
        steps: int
            The number of prediction candlestick steps that will be used in order to generate the data.
        price_change_requirement: float
            Price change percentage that will determine if the price moved up or down after a position is opened
        regressions: List[RegressionModel]
            The list of RegressionModels that will be used to generate the training data.
        include_rsi: bool
            If enabled, the RSI will be added as a feature with the column name "RSI".
        include_aroon: bool
            If enabled, the Aroon will be added as a feature with the column name "AROON".
        features_num: int
            The total number of features that will be used by the model to predict.
        df: DataFrame
            Pandas Dataframe containing all the features and labels populated every time a position
            is closed. This DF will be dumped as a csv once the process completes.
        active: Union[ITrainingDataActivePosition, None]
            Dictionary containing all the RegressionModel predictions as well as the up and down price
            details.
    """





    ## Initialization ##



    def __init__(self, config: ITrainingDataConfig, test_mode: bool = False):
        """Initializes the Training Data Instance as well as the candlesticks.

        Args:
            config: ITrainingDataConfig
                The configuration that will be used to generate the training data.
            test_mode: bool
                Indicates if the execution is running from unit tests.

        Raises:
            ValueError:
                If less than 1 RegressionModels are provided.
                If a duplicate Model ID is found.
                If a provided model isn't RegressionModel
        """
        # Make sure that at least 1 Regression Models were provided
        if len(config["regressions"]) < 1:
            raise ValueError(f"A minimum of 1 RegressionModels are required in order to generate \
                the classification training data. Received: {len(config['regressions'])}")

        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the Regression Selection
        self.regression_selection_id: str = config["regression_selection_id"]

        # Initialize the description
        self.description: str = config["description"]

        # Initialize the data that will be populated
        self.id: str = Utils.generate_uuid4()
        self.regressions: List[RegressionModel] = []
        df_data: Dict[str, List[float]] = {}
        ids: List[str] = []
        lookbacks: List[int] = []

        # Iterate over each Arima Model
        for m in config["regressions"]:
            # Make sure it isn"t a duplicate
            if m["id"] in ids:
                raise ValueError(f"Duplicate Model ID provided: {m['id']}")

            # Add the initialized model to the list
            self.regressions.append(RegressionModelFactory(m, True))
            
            # Populate helpers
            df_data[m["id"]] = []
            ids.append(m["id"])
            lookbacks.append(self.regressions[-1].get_lookback())

        # Initialize the max lookback
        self.max_lookback: int = max(lookbacks)

        # Initialize the candlesticks if not unit testing
        if not self.test_mode:
            Candlestick.init(self.max_lookback, Epoch.START, Epoch.END)

        # Init the start and end
        self.start: int = int(Candlestick.DF.iloc[0]["ot"])
        self.end: int = int(Candlestick.DF.iloc[-1]["ct"])

        # Init the steps
        self.steps: int = config["steps"]

        # Postitions Up & Down percent change requirements
        self.price_change_requirement: float = config["price_change_requirement"]

        # Init the Technical Analysis
        self.include_rsi: bool = config["include_rsi"]
        self.include_aroon: bool = config["include_aroon"]

        # Init the number of features
        self.features_num: int = self._get_features_num()

        # Complete the initial df
        self.df: DataFrame = self._complete_initial_df(df_data)

        # Initialize the active position
        self.active: Union[ITrainingDataActivePosition, None] = None






    def _get_features_num(self) -> int:
        """Calculates the total number of features that will be used by the
        regressions trained with the data that will be generated.

        Returns:
            int
        """
        # Init the base number of features
        features_num: int = len(self.regressions)

        # Check if the RSI is enabled
        if self.include_rsi:
            features_num += 1

        # Check if Aroon is enabled
        if self.include_aroon:
            features_num += 1

        # Finally, return the final number
        return features_num






    def _complete_initial_df(self, df_data: Dict[str, List[float]]) -> DataFrame:
        """Given a dict containing regression features, it will add any extra features
        as well as the labels and return the df.

        Args:
            df_data: Dict[str, List[float]]
                The dict containing regression ids as keys.

        Returns:
            DataFrame
        """
        # Add the TA Features if any
        if self.include_rsi:
            df_data["RSI"] = []
        if self.include_aroon:
            df_data["AROON"] = []

        # Add the labels
        df_data["up"] = []
        df_data["down"] = []

        # Finally, return the DataFrame
        return DataFrame(data=df_data)










    ## Execution ##




    def run(self) -> None:
        """Runs the training data execution based on the mode provided in the
        configuration.
        """
        # Check if it is a stepped execution
        if self.steps > 0:
            self.run_stepped()

        # Otherwise, run the traditional execution
        else:
            self.run_traditional()






    # Stepped Training Data
    # Iterates over stepped 1 minute candlesticks based on the prediction candlesticks' interval
    # minutes. The candlesticks between position open and close are not ignored. Instead, when
    # a position is closed, it goes back to where it left off.
    # The purpose of this type of training data is to generate a larger dataset for regressions to 
    # understand the relationship between features better.


    def run_stepped(self) -> None:
        """Runs the Stepped Training Data Process and stores the results once it completes.
        """
        # Calculate the number of 1 minute candlesticks that will be stepped
        real_steps: int = self.steps * Candlestick.PREDICTION_CANDLESTICK_CONFIG["interval_minutes"]

        # Init the progress bar
        progress_bar = tqdm(bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=ceil(Candlestick.DF.shape[0]/real_steps))

        # Init the time in which the execution started
        execution_start: int = Utils.get_time()

        # Iterate over the candlesticks based on the real steps
        for index in range(0, Candlestick.DF.shape[0]-1, real_steps):
            # Open a position
            self._open_position(Candlestick.DF.iloc[index])

            # Iterate until the position's outcome has been determined
            outcome_index: int = index + 1
            while self.active != None and outcome_index < Candlestick.DF.shape[0]:
                # Check the position
                self._check_position(Candlestick.DF.iloc[outcome_index])

                # Increment the index
                outcome_index += 1

            # Update the progress bar
            progress_bar.update()

        # Save the results
        Epoch.FILE.save_classification_training_data(self._build_file(execution_start))











    # Traditional Training Data
    # Iterates over the 1 minute candlesticks one by one opening positions whenever possible.
    # The candlesticks between a position open and close are ignored, similar to what real life
    # trading would be like.


    def run_traditional(self) -> None:
        """Runs the Training Data Process and stores the results once it completes.
        """
        # Init the progress bar
        progress_bar = tqdm(bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", total=Candlestick.DF.shape[0])

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
        Epoch.FILE.save_classification_training_data(self._build_file(execution_start))












    ## Positions Management ##



    def _open_position(self, candlestick: Series) -> None:
        """Creates a new position based on the open_price and the Models Predictions.

        Args:
            candlestick: Series
                The current candlestick.
        """
        # Calculate the position range
        up_price, down_price = self._get_position_range(candlestick["o"])

        # Build the features
        features: Dict[str, float] = self._get_features(candlestick["ot"])

        # Finally, populate the active position dict
        self.active = {
            "up_price": up_price,
            "down_price": down_price,
            "row": features
        }





    def _get_features(self, open_time: int) -> Dict[str, Union[int, float]]:
        """Retrieves the features that will be used by the model to predict.

        Args:
            open_time: int
        
        Returns:
            Dict[str, Union[int, float]]
            {[feature_name: str]: float}
        """
        # Init the lookback_df
        lookback_df: DataFrame = Candlestick.get_lookback_df(self.max_lookback, open_time)

        # Generate the Models' predictions
        features: Dict[str, Union[int, float]] = {
            m.id: m.feature(open_time, lookback_df=lookback_df) for m in self.regressions
        }

        # Check if any Technical Anlysis feature needs to be added
        if self.include_rsi or self.include_aroon:
            # Retrieve the technical analysis
            ta: ITechnicalAnalysis = TechnicalAnalysis.get_technical_analysis(
                lookback_df,
                include_rsi=self.include_rsi,
                include_aroon=self.include_aroon
            )

            # Populate the RSI feature if enabled
            if self.include_rsi:
                features["RSI"] = ta["rsi"]

            # Populate the Aroon feature if enabled
            if self.include_aroon:
                features["AROON"] = ta["aroon"]

        # Finally, return the features
        return features








    def _get_position_range(self, open_price: float) -> Tuple[float, float]:
        """Calculates the prices that will be used to determine if the 
        price went up or down.

        Args:
            open_price: float
                The current candlestick's open price.
        
        Returns:
            Tuple[float, float] (up_price, down_price)
        """
        return Utils.alter_number_by_percentage(open_price, self.price_change_requirement), \
            Utils.alter_number_by_percentage(open_price, -(self.price_change_requirement))








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


















    ## Results ## 






    def _build_file(self, execution_start: int) -> ITrainingDataFile:
        """Builds the Training Data File containing all the data collected.

        Args:
            execution_start: int
                The time in which the execution started.

        Returns:
            ITrainingDataFile
        """
        # Initialize the current time
        current_time: int = Utils.get_time()

        # Retrieve the price outcomes
        increase_outcomes, decrease_outcomes = self._get_price_outcomes()

        # Return the File Data
        return {
            "regression_selection_id": self.regression_selection_id,
            "id": self.id,
            "description": self.description,
            "creation_start": execution_start,
            "creation_end": current_time,
            "start": self.start,
            "end": self.end,
            "steps": self.steps,
            "price_change_requirement": self.price_change_requirement,
            "regressions": [m.get_model() for m in self.regressions],
            "include_rsi": self.include_rsi,
            "include_aroon": self.include_aroon,
            "features_num": self.features_num,
            "increase_outcome_num": increase_outcomes,
            "decrease_outcome_num": decrease_outcomes,
            "dataset_summary": self.df.describe().to_dict(),
            "training_data": ClassificationTrainingData.compress_training_data(self.df)
        }


    




    def _get_price_outcomes(self) -> Tuple[int, int]:
        """Retrieves the number of increase and decrease outcomes

        Returns:
            Tuple[int, int]
            (increase_outcomes, decrease_outcomes)
        """
        # Init the value counts
        up_count: Series = self.df["up"].value_counts()
        down_count: Series = self.df["down"].value_counts()

        # Return the packed values
        return int(up_count[1]) if up_count.get(1) is not None else 0,\
            int(down_count[1]) if down_count.get(1) is not None else 0










    

    ## Misc Helpers ## 




    @staticmethod
    def compress_training_data(df: DataFrame) -> ICompressedTrainingData:
        """Breaks down the training data DataFrame into a list of columns and rows.

        Args:
            df: DataFrame
                The df to be compressed.

        Returns:
            ICompressedTrainingData
        """
        return {"columns": df.columns.values.tolist(), "rows": df.values.tolist()}




    @staticmethod
    def decompress_training_data(data: ICompressedTrainingData) -> DataFrame:
        """Given a compressed training data dict, it will convert it into a DataFrame.

        Args:
            data: ICompressedTrainingData
                The data to be decompressed.

        Returns:
            DataFrame
        """
        return DataFrame(data=data["rows"], columns=data["columns"])





    @staticmethod
    def get_training_data_summary(file: ITrainingDataFile, train_size: int, test_size: int) -> ITrainingDataSummary:
        """Returns a brief overview of a Training Data File.

        Args:
            file: ITrainingDataFile
                The file generated by the training data execution.

        Returns:
            ITrainingDataSummary
        """
        return {
            "regression_selection_id": file["regression_selection_id"],
            "id": file["id"],
            "description": file["description"],
            "start": file["start"],
            "end": file["end"],
            "train_size": train_size,
            "test_size": test_size,
            "steps": file["steps"],
            "price_change_requirement": file["price_change_requirement"],
            "include_rsi": file["include_rsi"],
            "include_aroon": file["include_aroon"],
            "features_num": file["features_num"]
        }