from typing import List
from unittest import TestCase, main
from modules.types import IPrediction
from modules.database.Database import Database
from modules.candlestick.Candlestick import Candlestick
from modules.prediction_cache.TemporaryPredictionCache import TemporaryPredictionCache


## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





## Test Data ##
LOOKBACK: int = 100
PREDS: List[IPrediction] = [
    {'r': 1,'t': 1650654480000,'md': [{'d': 'long','pl': [11545.65, 11611.85, 11635.41, 11666.99, 11785.14]}]},
    {'r': -1,'t': 1650644640000,'md': [{'d': 'short','pl': [11545.65, 11525.85, 11510.63, 11485.36, 11320.55, 11318.63, 11654,11, 10854.01, 10594, 10574.54]}]},
    {'r': 0,'t': 1650633120000,'md': [{'d': 'neutral','pl': [11545.65, 11541.36, 11542.84, 11547.65, 11529.85]}]},
    {'r': 1,'t': 1603370040000,'md': [{'d': 'long','npl': [0.564527, 0.564006, 0.560657, 0.551291, 0.555746, 0.553255]}]},
    {'r': -1,'t': 1611994740000,'md': [{'d': 'short','npl': [0.021826, 0.022541, 0.022473, 0.022549, 0.022775, 0.023325]}]},
    {'r': 1,'t': 1622163300000,'md': [{'d': 'long','f': [1.0,1.0,0.0,-1.0,-1.0], "up": 0.5965415, "dp": 0.4112145}]},
    {'r': 0,'t': 1630973940000,'md': [{'d': 'neutral'}]}
]







# Test Class
class TemporaryPredictionCacheTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass





    # Saves and validates a series of predictions
    def testCacheFlow(self):
        # Init the cache
        cache: TemporaryPredictionCache = TemporaryPredictionCache()

        # There should be no items stored
        self.assertEqual(len(cache.predictions.keys()), 0)

        # Iterate over each pred
        for pred in PREDS:
            # Retrieve the lookback prediction range
            first_ot, last_ct = Candlestick.get_lookback_prediction_range(LOOKBACK, pred['t'])

            # Make sure it does not exist
            self.assertEqual(cache.get(first_ot, last_ct), None)

            # Save it
            cache.save(first_ot, last_ct, pred)

            # Make sure it matches the original
            self.assertDictEqual(cache.get(first_ot, last_ct), pred)

        # Make sure all predictions are still stored
        self.assertEqual(len(cache.predictions.keys()), len(PREDS))



            








# Test Execution
if __name__ == '__main__':
    main()