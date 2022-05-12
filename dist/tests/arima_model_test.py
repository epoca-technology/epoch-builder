import unittest
from typing import Union
from copy import deepcopy
from modules.candlestick import Candlestick
from modules.database import get_pred, delete_pred
from modules.model import ArimaModel, IModel, IPrediction


## Test Model ##
BASIC_CONFIG: IModel = {
    'id': 'A212',
    'arima_models': [{
        'lookback': 150,
        'predictions': 10,
        'arima': { 'p': 2, 'd': 1, 'q': 2 },
        'interpreter': { 'long': 0.5, 'short': 0.5 }
    }]
}


## Current Timestamp ##
CURRENT_TIME: int = Candlestick.DF.iloc[1585]['ot']



## Test Class ##
class ArimaModelTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    ## Initialization ##



    # Can initialize a model with default values
    def testInitWithDefaultValues(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)
        del config['arima_models'][0]['lookback']
        del config['arima_models'][0]['predictions']
        del config['arima_models'][0]['interpreter']

        # Init the model
        model: ArimaModel = ArimaModel(config)

        # Make sure the ID is correct
        self.assertEqual(config['id'], model.id)

        # Make sure the lookback is correct
        self.assertEqual(model.get_lookback(), ArimaModel.DEFAULT_LOOKBACK)

        # Retrieve the summary and make sure there is a full match
        summary: IModel = model.get_model()

        # Make sure the summary includes only 1 model
        self.assertEqual(len(summary['arima_models']), 1)

        # Complete the config dict with the default values the model would've inserted
        config['arima_models'][0]['arima'] = { 'p': 2, 'd': 1, 'q': 2, 'P': 0, 'D': 0, 'Q': 0, 'm': 0 }
        config['arima_models'][0]['lookback'] = ArimaModel.DEFAULT_LOOKBACK
        config['arima_models'][0]['predictions'] = ArimaModel.DEFAULT_PREDICTIONS
        config['arima_models'][0]['interpreter'] = ArimaModel.DEFAULT_INTERPRETER

        # Make sure there is a perfect match
        self.assertDictEqual(summary['arima_models'][0], config['arima_models'][0])




    # Can initialize a model with custom values
    def testInitWithCustomValues(self):
        # Init the config
        config: IModel = {
            'id': 'A3233234',
            'arima_models': [{
                'lookback': 250,
                'predictions': 15, 
                'arima': { 'p': 3, 'd': 2, 'q': 3, 'P': 3, 'D': 2, 'Q': 3, 'm': 4 },
                'interpreter': { 'long': 0.7, 'short': 0.7 }
            }]
        }

        # Init the model
        model: ArimaModel = ArimaModel(config)

        # Make sure the ID is correct
        self.assertEqual(config['id'], model.id)

        # Make sure the max_lookback is correct
        self.assertEqual(model.get_lookback(), config['arima_models'][0]['lookback'])

        # Retrieve the summary and make sure there is a full match
        summary: IModel = model.get_model()

        # Make sure the summary is valid
        self.assertEqual(len(summary['arima_models']), 1)
        self.assertDictEqual(summary['arima_models'][0], config['arima_models'][0])






    ## Predictions



    # Can perform predictions with a basic config
    def testBasicPrediction(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, enable_cache=False)

        # Validate the integrity of the result
        self.assertIsInstance(pred['r'], int)
        self.assertTrue(pred['r'] == -1 or pred['r'] == 0 or pred['r'] == 1)

        # Validate the integrity of the time
        self.assertEqual(pred['t'], CURRENT_TIME)
        
        # Validate the integrity of the metadata
        self.assertIsInstance(pred['md'], list)
        self.assertEqual(len(pred['md']), 1)
        self.assertIsInstance(pred['md'][0], dict)
        self.assertIsInstance(pred['md'][0]['pl'], list)
        self.assertEqual(len(pred['md'][0]['pl']), m.predictions)
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)





    # Can predict without storing data in cache
    def testPredictWithoutCache(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)

        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, enable_cache=False)

        # When not using cache, the prediction's metadata should not be minimized
        self.assertIsInstance(pred['md'][0].get('pl'), list)

        # Retrieve the prediction range
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(m.get_lookback(), CURRENT_TIME)

        # Make sure the prediction was not stored
        cached_pred: Union[IPrediction, None] = get_pred(m.id, first_ot, last_ct)
        self.assertEqual(cached_pred, None)





    # Can predict storing data in cache
    def testPredictWithCache(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)

        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, enable_cache=True)

        # When using cache, the prediction's metadata should be minimized
        self.assertEqual(pred['md'][0].get('pl'), None)

        # Retrieve the prediction range
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(m.get_lookback(), CURRENT_TIME)

        # Make sure the prediction was stored
        cached_pred: Union[IPrediction, None] = get_pred(m.id, first_ot, last_ct)
        self.assertFalse(cached_pred == None)
        self.assertDictEqual(pred, cached_pred)
        self.assertEqual(cached_pred['md'][0].get('pl'), None)

        # Clean up the prediction
        delete_pred(m.id, first_ot, last_ct)
        cached_pred = get_pred(m.id, first_ot, last_ct)
        self.assertTrue(cached_pred == None)






    # Can perform a Sarima prediction and store it in cache
    def testPredictSarimaWithCache(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)
        config['id'] = "A2221116"
        config['arima_models'][0]['arima'] = { 'p': 2, 'd': 2, 'q': 2, 'P': 1, 'D': 1, 'Q': 1, 'm': 6 }

        # Initialize the model
        m = ArimaModel(config)

        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, enable_cache=True)

        # When using cache, the prediction's metadata should be minimized
        self.assertEqual(pred['md'][0].get('pl'), None)

        # Retrieve the prediction range
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(m.get_lookback(), CURRENT_TIME)

        # Make sure the prediction was stored
        cached_pred: Union[IPrediction, None] = get_pred(m.id, first_ot, last_ct)
        self.assertFalse(cached_pred == None)
        self.assertDictEqual(pred, cached_pred)
        self.assertEqual(cached_pred['md'][0].get('pl'), None)

        # Clean up the prediction
        delete_pred(m.id, first_ot, last_ct)
        cached_pred = get_pred(m.id, first_ot, last_ct)
        self.assertTrue(cached_pred == None)






    # Can validate the integrity of a model when it follows the Apdq guideline
    def testValidateIntegrity(self):
        # Can initialize a valid model
        ArimaModel({
            'id': 'A314',
            'arima_models': [{
                'lookback': 150,
                'predictions': 10, 
                'arima': { 'p': 3, 'd': 1, 'q': 4 },
                'interpreter': {'long': 0.5, 'short': 0.5 }
            }]
        })

        # Cannot initialize a model with an invalid integrity
        with self.assertRaises(ValueError):
            ArimaModel({
                'id': 'A314',
                'arima_models': [{
                    'lookback': 150,
                    'predictions': 10, 
                    'arima': { 'p': 3, 'd': 1, 'q': 6 },
                    'interpreter': { 'long': 0.5, 'short': 0.5 }
                }]
            })




# Test Execution
if __name__ == '__main__':
    unittest.main()