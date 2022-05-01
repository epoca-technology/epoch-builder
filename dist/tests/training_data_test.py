import unittest
from typing import List
from copy import deepcopy
from pandas import Series
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.model import TrainingData, ITrainingDataConfig




## Test Helpers

# Config
DEFAULT_CONFIG: ITrainingDataConfig = {
    'start': None, 
    'end': None, 
    'up_percent_change': 1, 
    'down_percent_change': 1, 
    'single_models': [
        {
            'id': 'A101',
            'single_models': [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 1, 'd': 0, 'q': 1 },
                'interpreter': { 'long': 0.5, 'short': 0.5 }
            }]
        },
        {
            'id': 'A111',
            'single_models': [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 1, 'd': 1, 'q': 1 },
                'interpreter': { 'long': 0.5, 'short': 0.5 }
            }]
        },
        {
            'id': 'A112',
            'single_models': [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 1, 'd': 1, 'q': 2 },
                'interpreter': { 'long': 0.5, 'short': 0.5 }
            }]
        },
        {
            'id': 'A121',
            'single_models': [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 1, 'd': 2, 'q': 1 },
                'interpreter': { 'long': 0.5, 'short': 0.5 }
            }]
        },
        {
            'id': 'A211',
            'single_models': [{
                'lookback': 300,
                'arima': { 'predictions': 10, 'p': 2, 'd': 1, 'q': 1 },
                'interpreter': { 'long': 0.5, 'short': 0.5 }
            }]
        },
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
        td: TrainingData = TrainingData(config, test_mode=True)

        # Make sure the models have been initialized correctly
        self.assertEqual(len(td.single_models), len(config['single_models']))
        for i, sm in enumerate(config['single_models']):
            if sm['id'] != td.single_models[i].id:
                self.fail(f"Single Model ID Missmatch: {sm['id']} != {td.single_models[i].id}")

        # Make sure the ID was generated successfully
        expected_id: str = ''
        for sm in config['single_models']:
            expected_id = expected_id + sm['id']
        self.assertEqual(expected_id, td.id)

        # Make sure the dates have been set correctly
        self.assertEqual(td.start, int(Candlestick.DF.iloc[0]['ot']))
        self.assertEqual(td.end, int(Candlestick.DF.iloc[-1]['ct']))

        # Make sure the position percentages have been set correctly
        self.assertEqual(config['up_percent_change'], td.up_percent_change)
        self.assertEqual(config['down_percent_change'], td.down_percent_change)

        # Validate the integrity of the DF
        self.assertEqual(td.df.shape[0], 0)
        self.assertEqual(td.df.shape[1], len(config['single_models']) + 2)
        


    # Cannot initialize with less than 5 single models
    def testInitializeWithLessThan5Models(self):
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        config['single_models'] = config['single_models'][0:3]
        with self.assertRaises(ValueError):
            TrainingData(config, test_mode=True)


    # Cannot initialize with duplicate single models
    def testInitializeWithDuplicateModels(self):
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        config['single_models'][3] = config['single_models'][4]
        with self.assertRaises(ValueError):
            TrainingData(config, test_mode=True)

    # Cannot initialize with different single models lookbacks
    def testInitializeWithDifferentLookbacks(self):
        config: ITrainingDataConfig = deepcopy(DEFAULT_CONFIG)
        config['single_models'][3]['single_models'][0]['lookback'] = 280
        with self.assertRaises(ValueError):
            TrainingData(config, test_mode=True)







    ## Positions ## 


 










    ## Results ##










# Test Execution
if __name__ == '__main__':
    unittest.main()
