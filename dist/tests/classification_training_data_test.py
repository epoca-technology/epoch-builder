import unittest
from typing import List
from copy import deepcopy
from pandas import Series, DataFrame
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.classification import ClassificationTrainingData, ITrainingDataConfig, ICompressedTrainingData, \
    compress_training_data, decompress_training_data




## Test Helpers

# Config
DEFAULT_CONFIG: ITrainingDataConfig = {
    "description": "Unit Test",
    "start": "22/03/2022", 
    "end": "22/04/2022", 
    'up_percent_change': 2, 
    'down_percent_change': 2, 
    'models': [
        { "id": "A101","arima_models": [{"arima": {"p": 1, "d": 0,"q": 1}}] },
        { "id": "A111","arima_models": [{"arima": {"p": 1, "d": 1,"q": 1}}] },
        { "id": "A112","arima_models": [{"arima": {"p": 1, "d": 1,"q": 2}}] },
        { "id": "A121","arima_models": [{"arima": {"p": 1, "d": 2,"q": 1}}] },
        { "id": "R_UNIT_TEST","regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": {"long": 1.5, "short": 1.5}}] }
    ]
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
        self.assertEqual(td.description, config['description'])

        # Make sure the dates have been set correctly
        self.assertEqual(td.start, int(Candlestick.DF.iloc[0]['ot']))
        self.assertEqual(td.end, int(Candlestick.DF.iloc[-1]['ct']))

        # Make sure the position percentages have been set correctly
        self.assertEqual(config['up_percent_change'], td.up_percent_change)
        self.assertEqual(config['down_percent_change'], td.down_percent_change)

        # Validate the integrity of the DF
        self.assertEqual(td.df.shape[0], 0)
        self.assertEqual(td.df.shape[1], len(config['models']) + 2)
        


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







    ## Positions ## 


 










    ## Results ##











    ## Training Data Compression ##



    # Can compress and decompress training data
    def testCompressAndDecompressTrainingData(self):
        # Initialize the DataFrame
        df = DataFrame(data={
            DEFAULT_CONFIG["models"][0]["id"]: [1, 2, 1, 0, 2],
            DEFAULT_CONFIG["models"][1]["id"]: [2, 1, 0, 0, 1],
            DEFAULT_CONFIG["models"][2]["id"]: [1, 2, 2, 2, 0],
            DEFAULT_CONFIG["models"][3]["id"]: [2, 1, 2, 2, 1],
            DEFAULT_CONFIG["models"][4]["id"]: [0, 0, 1, 2, 1],
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
            [1, 2, 1, 2, 0, 1, 0],
            [2, 1, 2, 1, 0, 0, 1],
            [1, 0, 2, 2, 1, 0, 1],
            [0, 0, 2, 2 ,2, 1, 0],
            [2, 1, 0, 1 ,1 ,1, 0]
        ])

        # Decompress the data and validate its integrity
        decompressed: DataFrame = decompress_training_data(compressed)
        self.assertTrue(df.equals(decompressed))







# Test Execution
if __name__ == '__main__':
    unittest.main()
