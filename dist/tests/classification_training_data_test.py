import unittest
from typing import List
from copy import deepcopy
from pandas import Series, DataFrame
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.classification import ClassificationTrainingData, ITrainingDataConfig, ICompressedTrainingData, \
    ITrainingDataFile, compress_training_data, decompress_training_data




## Test Helpers

# Default Configuration
DEFAULT_CONFIG: ITrainingDataConfig = {
    "regression_selection_id": "e5a03686-7bb9-4e2f-ab2f-3058281f589f",
    "description": "UNIT_TEST: DO NOT DELETE.",
    "start": "22/03/2022", 
    "end": "22/04/2022", 
    "steps": 0,
    'up_percent_change': 2, 
    'down_percent_change': 2, 
    'models': [
        { "id": "A101","arima_models": [{"arima": {"p": 1, "d": 0,"q": 1}}] },
        { "id": "A111","arima_models": [{"arima": {"p": 1, "d": 1,"q": 1}}] },
        { "id": "A112","arima_models": [{"arima": {"p": 1, "d": 1,"q": 2}}] },
        { "id": "A121","arima_models": [{"arima": {"p": 1, "d": 2,"q": 1}}] },
        { "id": "R_UNIT_TEST","regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": {"long": 1.5, "short": 1.5}}] }
    ],
    "include_rsi": False,
    "include_aroon": False
}







# Next Candlestick Retriever
def _get_next(candlestick: Series, change: float) -> Series:
    """Given a candlestick, it will fabricate the next given.

    Args:
        candlestick: Series
            The candlestick to base the new one on.
        change: float
            The change that will be applied to the prices.
    
    Returns:
        Series
    """
    c = candlestick.copy()
    c['ot'] = Utils.add_minutes(candlestick['ot'], 1)
    c['ct'] = Utils.add_minutes(candlestick['ot'], 2) - 1
    c['o'] = Utils.alter_number_by_percentage(candlestick['o'], change)
    c['h'] = Utils.alter_number_by_percentage(candlestick['h'], change)
    c['l'] = Utils.alter_number_by_percentage(candlestick['l'], change)
    c['c'] = Utils.alter_number_by_percentage(candlestick['c'], change)
    return c



# Start
START_CANDLESTICK: Series = Candlestick.DF.iloc[35888].copy()
START_CANDLESTICK['o'] = 40050.85
START_CANDLESTICK['h'] = 40068.52
START_CANDLESTICK['l'] = 40046.37
START_CANDLESTICK['c'] = 40052.18





## Test Class ##
class TrainingDataTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    ## Initialization ##


    # Initialize an instance with valid data and validate the integrity
    def testInitialize(self):
        # Init the config
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)

        # Init the instance
        td: ClassificationTrainingData = ClassificationTrainingData(config, test_mode=True)

        # Make sure the models have been initialized correctly
        self.assertEqual(len(td.models), len(config['models']))
        for i, m in enumerate(config['models']):
            if m['id'] != td.models[i].id:
                self.fail(f"Model ID Missmatch: {m['id']} != {td.models[i].id}")

        # Make sure the ID and the description were initialized
        self.assertTrue(Utils.is_uuid4(td.id))
        self.assertEqual(td.regression_selection_id, config['regression_selection_id'])
        self.assertEqual(td.description, config['description'])

        # Make sure the dates have been set correctly
        self.assertEqual(td.start, int(Candlestick.DF.iloc[0]['ot']))
        self.assertEqual(td.end, int(Candlestick.DF.iloc[-1]['ct']))

        # Make sure the steps have been set correctly
        self.assertEqual(config['steps'], td.steps)

        # Make sure the position percentages have been set correctly
        self.assertEqual(config['up_percent_change'], td.up_percent_change)
        self.assertEqual(config['down_percent_change'], td.down_percent_change)

        # Make sure the TA has been set correctly
        self.assertEqual(config['include_rsi'], td.include_rsi)
        self.assertEqual(config['include_aroon'], td.include_aroon)

        # Make sure the Features number has been set correctly
        self.assertEqual(len(config['models']), td.features_num)

        # Validate the integrity of the DF
        self.assertEqual(td.df.shape[0], 0)
        self.assertEqual(td.df.shape[1], len(config['models']) + 2)
        for i, column_name in enumerate(td.df.columns):
            if column_name != "up" and column_name != "down":
                self.assertEqual(column_name, config["models"][i]["id"])
        





    # Initialize an instance with valid data (Including the RSI) and validate the integrity
    def testInitializeWithRSI(self):
        # Init the config
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)

        # Set the RSI value
        config["include_rsi"] = True

        # Init the instance
        td: ClassificationTrainingData = ClassificationTrainingData(config, test_mode=True)

        # Make sure the models have been initialized correctly
        self.assertEqual(len(td.models), len(config['models']))
        for i, m in enumerate(config['models']):
            if m['id'] != td.models[i].id:
                self.fail(f"Model ID Missmatch: {m['id']} != {td.models[i].id}")

        # Make sure the ID and the description were initialized
        self.assertTrue(Utils.is_uuid4(td.id))
        self.assertEqual(td.regression_selection_id, config['regression_selection_id'])
        self.assertEqual(td.description, config['description'])

        # Make sure the dates have been set correctly
        self.assertEqual(td.start, int(Candlestick.DF.iloc[0]['ot']))
        self.assertEqual(td.end, int(Candlestick.DF.iloc[-1]['ct']))

        # Make sure the steps have been set correctly
        self.assertEqual(config['steps'], td.steps)

        # Make sure the position percentages have been set correctly
        self.assertEqual(config['up_percent_change'], td.up_percent_change)
        self.assertEqual(config['down_percent_change'], td.down_percent_change)

        # Make sure the TA has been set correctly
        self.assertEqual(config['include_rsi'], td.include_rsi)
        self.assertEqual(config['include_aroon'], td.include_aroon)

        # Make sure the Features number has been set correctly
        self.assertEqual(len(config['models']) + 1, td.features_num)

        # Validate the integrity of the DF
        self.assertEqual(td.df.shape[0], 0)
        self.assertEqual(td.df.shape[1], len(config['models']) + 1 + 2)
        rsi_column_exists: bool = False
        for i, column_name in enumerate(td.df.columns):
            if column_name == "RSI":
                rsi_column_exists = True
            if  column_name != "RSI" and column_name != "up" and column_name != "down":
                self.assertEqual(column_name, config["models"][i]["id"])
        self.assertTrue(rsi_column_exists)




    # Initialize an instance with valid data (Including Multiple TA Indicators) and validate the integrity
    def testInitializeWithMultipleIndicators(self):
        # Init the config
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)

        # Set the RSI value
        config["include_rsi"] = True
        config["include_aroon"] = True

        # Init the instance
        td: ClassificationTrainingData = ClassificationTrainingData(config, test_mode=True)

        # Make sure the models have been initialized correctly
        self.assertEqual(len(td.models), len(config['models']))
        for i, m in enumerate(config['models']):
            if m['id'] != td.models[i].id:
                self.fail(f"Model ID Missmatch: {m['id']} != {td.models[i].id}")

        # Make sure the ID and the description were initialized
        self.assertTrue(Utils.is_uuid4(td.id))
        self.assertEqual(td.regression_selection_id, config['regression_selection_id'])
        self.assertEqual(td.description, config['description'])

        # Make sure the dates have been set correctly
        self.assertEqual(td.start, int(Candlestick.DF.iloc[0]['ot']))
        self.assertEqual(td.end, int(Candlestick.DF.iloc[-1]['ct']))

        # Make sure the steps have been set correctly
        self.assertEqual(config['steps'], td.steps)

        # Make sure the position percentages have been set correctly
        self.assertEqual(config['up_percent_change'], td.up_percent_change)
        self.assertEqual(config['down_percent_change'], td.down_percent_change)

        # Make sure the TA has been set correctly
        self.assertEqual(config['include_rsi'], td.include_rsi)
        self.assertEqual(config['include_aroon'], td.include_aroon)

        # Make sure the Features number has been set correctly
        self.assertEqual(len(config['models']) + 3, td.features_num)

        # Validate the integrity of the DF
        self.assertEqual(td.df.shape[0], 0)
        self.assertEqual(td.df.shape[1], len(config['models']) + 3 + 2)
        rsi_column_exists: bool = False
        aroon_up_column_exists: bool = False
        aroon_down_column_exists: bool = False
        for i, column_name in enumerate(td.df.columns):
            if column_name == "RSI":
                rsi_column_exists = True
            if column_name == "AROON_UP":
                aroon_up_column_exists = True
            if column_name == "AROON_DOWN":
                aroon_down_column_exists = True
            if  column_name != "RSI" and column_name != "AROON_UP" and column_name != "AROON_DOWN" \
                and column_name != "up" and column_name != "down":
                self.assertEqual(column_name, config["models"][i]["id"])
        self.assertTrue(rsi_column_exists)
        self.assertTrue(aroon_up_column_exists)
        self.assertTrue(aroon_down_column_exists)





        


    # Cannot initialize with less than 5 Models
    def testInitializeWithLessThan5Models(self):
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        config['models'] = config['models'][0:3]
        with self.assertRaises(ValueError):
            ClassificationTrainingData(config, test_mode=True)


    # Cannot initialize with duplicate Models
    def testInitializeWithDuplicateModels(self):
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        config['models'][3] = config['models'][4]
        with self.assertRaises(ValueError):
            ClassificationTrainingData(config, test_mode=True)









    ## Positions Flow ## 


 
    # Can calculate the correct position range
    def testPositionRange(self):
        # Init the Training Data Instance
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        td: ClassificationTrainingData = ClassificationTrainingData(config, test_mode=True)

        # Retrieve the range for a random open price
        price: float = 41865.96
        up_price, down_price = td._get_position_range(price)

        # Validate the up price
        self.assertEqual(up_price, Utils.alter_number_by_percentage(price, config["up_percent_change"]))

        # Validate the down price
        self.assertEqual(down_price, Utils.alter_number_by_percentage(price, -config["down_percent_change"]))







 
    # Can initialize a series of positions and handle them properly
    def testFullPositionsFlow(self):
        # Init the Training Data Instance
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        td: ClassificationTrainingData = ClassificationTrainingData(config, test_mode=True)

        # Init the candlesticks list
        candlesticks: List[Series] = [START_CANDLESTICK]

        # There shouldnt be an open position
        self.assertEqual(td.df.shape[0], 0)
        self.assertEqual(td.active, None)

        # Open a position
        td._open_position(candlesticks[-1])

        # Make sure the position was actually opened
        self.assertEqual(td.df.shape[0], 0)
        self.assertIsInstance(td.active, dict)
        self.assertEqual(td.active["up_price"], Utils.alter_number_by_percentage(candlesticks[-1]["o"], config["up_percent_change"]))
        self.assertEqual(td.active["down_price"], Utils.alter_number_by_percentage(candlesticks[-1]["o"], -config["down_percent_change"]))
        self.assertIsInstance(td.active["row"], dict)

        # A position check with a change that doesn't meet the up_percent_change does nothing
        candlesticks.append(_get_next(candlesticks[-1], round(config["up_percent_change"]/2, 2)))
        td._check_position(candlesticks[-1])
        self.assertEqual(td.df.shape[0], 0)
        self.assertIsInstance(td.active, dict)

        # Checking a position with a change that hits the up_percent_change the position will be closed as up
        candlesticks.append(_get_next(candlesticks[-1], config["up_percent_change"]))
        td._check_position(candlesticks[-1])
        self.assertEqual(td.df.shape[0], 1)
        self.assertEqual(td.active, None)
        self.assertEqual(td.df.iloc[0]["up"], 1)
        self.assertEqual(td.df.iloc[0]["down"], 0)

        # Open a new position
        candlesticks.append(_get_next(candlesticks[-1], 1.5))
        td._open_position(candlesticks[-1])

        # Make sure the position was actually opened
        self.assertEqual(td.df.shape[0], 1)
        self.assertIsInstance(td.active, dict)
        self.assertEqual(td.active["up_price"], Utils.alter_number_by_percentage(candlesticks[-1]["o"], config["up_percent_change"]))
        self.assertEqual(td.active["down_price"], Utils.alter_number_by_percentage(candlesticks[-1]["o"], -config["down_percent_change"]))
        self.assertIsInstance(td.active["row"], dict)

        # A position check with a change that doesn't meet the down_percent_change does nothing
        candlesticks.append(_get_next(candlesticks[-1], -round(config["down_percent_change"]/2, 2)))
        td._check_position(candlesticks[-1])
        self.assertEqual(td.df.shape[0], 1)
        self.assertIsInstance(td.active, dict)

        # Checking a position with a change that hits the down_percent_change the position will be closed as down
        candlesticks.append(_get_next(candlesticks[-1], -config["down_percent_change"]))
        td._check_position(candlesticks[-1])
        self.assertEqual(td.df.shape[0], 2)
        self.assertEqual(td.active, None)
        self.assertEqual(td.df.iloc[1]["up"], 0)
        self.assertEqual(td.df.iloc[1]["down"], 1)

        # Open a final position
        candlesticks.append(_get_next(candlesticks[-1], 2.85))
        td._open_position(candlesticks[-1])

        # Close the position as up and make sure it was closed
        candlesticks.append(_get_next(candlesticks[-1], round(config["up_percent_change"]*2, 2)))
        td._check_position(candlesticks[-1])
        self.assertEqual(td.df.shape[0], 3)
        self.assertEqual(td.active, None)
        self.assertEqual(td.df.iloc[2]["up"], 1)
        self.assertEqual(td.df.iloc[2]["down"], 0)

        # Build the file and validate its integrity
        file: ITrainingDataFile = td._build_file(Utils.get_time()-60000)

        # Validate basic values
        self.assertEqual(file["id"], td.id)
        self.assertEqual(file["description"], config["description"])
        self.assertEqual(file["up_percent_change"], config["up_percent_change"])
        self.assertEqual(file["down_percent_change"], config["down_percent_change"])

        # Validate the models
        self.assertEqual(len(file["models"]), len(config["models"]))
        for i, model in enumerate(config["models"]):
            self.assertEqual(model["id"], file["models"][i]["id"])

        # Validate the price action insights
        self.assertEqual(file["price_actions_insight"]["up"], 2)
        self.assertEqual(file["price_actions_insight"]["down"], 1)

        # Validate the predictions insights
        for model in config["models"]:
            self.assertIsInstance(file["predictions_insight"][model["id"]], dict)
            self.assertIsInstance(file["predictions_insight"][model["id"]]["long"], int)
            self.assertIsInstance(file["predictions_insight"][model["id"]]["short"], int)
            self.assertIsInstance(file["predictions_insight"][model["id"]]["neutral"], int)

        ## Validate the Training Data ##

        # The compressed training data should be a dict
        self.assertIsInstance(file["training_data"], dict)

        # The columns list must include all features and labels
        self.assertIsInstance(file["training_data"]["columns"], list)
        self.assertEqual(len(file["training_data"]["columns"]), len(config["models"]) + 2)
        self.assertListEqual(
            file["training_data"]["columns"],
            [m["id"] for m in config["models"]] + ["up", "down"]
        )

        # There should be 3 rows following the positions' outcomes (up, down, up) and predictions
        self.assertIsInstance(file["training_data"]["rows"], list)
        self.assertEqual(len(file["training_data"]["rows"]), 3)
        for row_index, row in enumerate(file["training_data"]["rows"]):
            for column_index, column in enumerate(td.df.columns):
                self.assertEqual(row[column_index], td.df.iloc[row_index][column])

        # Finally, decompress the training data and compare it to the original df
        self.assertTrue(td.df.equals(decompress_training_data(file["training_data"])))


















    ## Training Data Compression ##



    # Can compress and decompress training data
    def testCompressAndDecompressTrainingData(self):
        # Initialize the DataFrame
        df = DataFrame(data={
            DEFAULT_CONFIG["models"][0]["id"]: [-1, 1, -1, 0, 1],
            DEFAULT_CONFIG["models"][1]["id"]: [1, -1, 0, 0, -1],
            DEFAULT_CONFIG["models"][2]["id"]: [-1, 1, 1, 1, 0],
            DEFAULT_CONFIG["models"][3]["id"]: [1, -1, 1, 1, -1],
            DEFAULT_CONFIG["models"][4]["id"]: [0, 0, -1, 1, -1],
            "up":                              [1, 0, 0, 1, 1],
            "down":                            [0, 1, 1, 0, 0],
        })

        # Compress the training data and validate its integrity
        compressed: ICompressedTrainingData = compress_training_data(df)

        # Validate the columns
        columns: List[str] = [m["id"] for m in DEFAULT_CONFIG["models"]] + ["up", "down"]
        self.assertListEqual(compressed["columns"], columns)

        # Validate the rows
        self.assertListEqual(compressed["rows"], [
            [-1, 1, -1, 1, 0, 1, 0],
            [1, -1, 1, -1, 0, 0, 1],
            [-1, 0, 1, 1, -1, 0, 1],
            [0, 0, 1, 1, 1, 1, 0],
            [1, -1, 0, -1 ,-1 ,1, 0]
        ])

        # Decompress the data and validate its integrity
        decompressed: DataFrame = decompress_training_data(compressed)
        self.assertTrue(df.equals(decompressed))







# Test Execution
if __name__ == '__main__':
    unittest.main()
