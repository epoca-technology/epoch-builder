import unittest
from typing import Union
from copy import deepcopy
from modules.candlestick import Candlestick
from modules.database import get_prediction, delete_prediction
from modules.model import SingleModel, IModel, IPrediction


## Test Model ##
BASIC_CONFIG: IModel = {
    'id': 'TEST_MODEL',
    'single_models': [{
        'lookback': 150,
        'arima': { 'predictions': 10, 'p': 2, 'd': 1, 'q': 2 },
        'interpreter': {
            'long': 0.5, 
            'short': 0.5,
        }
    }]
}


## Current Timestamp ##
CURRENT_TIME: int = Candlestick.DF.iloc[1585]['ot']



## Test Class ##
class SingleModelTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    # Can initialize a model with default values
    def testInitWithDefaultValues(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Init the model
        model: SingleModel = SingleModel(config)

        # Make sure the max_lookback is correct
        self.assertEqual(model.get_max_lookback(), config['single_models'][0]['lookback'])

        # Retrieve the summary and make sure there is a full match
        summary: IModel = model.get_model()

        # Make sure the summary includes only 1 model
        self.assertEqual(len(summary['single_models']), 1)

        # Complete the config dict with the default values the model would've inserted
        config['single_models'][0]['arima'] = { 'predictions': 10, 'p': 2, 'd': 1, 'q': 2, 'P': 0, 'D': 0, 'Q': 0, 'm': 0 }
        config['single_models'][0]['interpreter']['rsi'] = {'active': False} 
        config['single_models'][0]['interpreter']['ema'] = {'active': False} 

        # Make sure there is a perfect match
        self.assertDictEqual(summary['single_models'][0], config['single_models'][0])




    # Can initialize a model with custom values
    def testInitWithCustomValues(self):
        # Init the config
        config: IModel = {
            'id': 'TEST_MODEL',
            'single_models': [{
                'lookback': 250,
                'arima': { 'predictions': 10, 'p': 3, 'd': 2, 'q': 3, 'P': 3, 'D': 2, 'Q': 3, 'm': 4 },
                'interpreter': {
                    'long': 0.7, 
                    'short': 0.7, 
                    'rsi': {'active': True, 'overbought': 70, 'oversold': 30},
                    'ema': {'active': True, 'distance': 0.3},
                }
            }]
        }

        # Init the model
        model: SingleModel = SingleModel(config)

        # Make sure the max_lookback is correct
        self.assertEqual(model.get_max_lookback(), config['single_models'][0]['lookback'])

        # Retrieve the summary and make sure there is a full match
        summary: IModel = model.get_model()

        # Make sure the summary is valid
        self.assertEqual(len(summary['single_models']), 1)
        self.assertDictEqual(summary['single_models'][0], config['single_models'][0])





    # Can perform predictions with a basic config
    def testBasicPrediction(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = SingleModel(config)
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME)

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
        self.assertEqual(len(pred['md'][0]['pl']), m.arima['predictions'])
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)
        self.assertEqual(pred['md'][0].get('rsi'), None)
        self.assertEqual(pred['md'][0].get('sema'), None)
        self.assertEqual(pred['md'][0].get('lema'), None)




    # Can perform predictions with RSI
    def testPredictionWithRSI(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)
        config['single_models'][0]['interpreter']['rsi'] = {'active': True}
        
        # Initialize the model
        m = SingleModel(config)

        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME)
        
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
        self.assertEqual(len(pred['md'][0]['pl']), m.arima['predictions'])
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)
        self.assertIsInstance(pred['md'][0]['rsi'], float)
        self.assertEqual(pred['md'][0].get('sema'), None)
        self.assertEqual(pred['md'][0].get('lema'), None)





    # Can perform predictions with EMA
    def testPredictionWithEMA(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)
        config['single_models'][0]['interpreter']['ema'] = {'active': True}

        # Initialize the model
        m = SingleModel(config)
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME)

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
        self.assertEqual(len(pred['md'][0]['pl']), m.arima['predictions'])
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)
        self.assertEqual(pred['md'][0].get('rsi'), None)
        self.assertIsInstance(pred['md'][0]['sema'], float)
        self.assertIsInstance(pred['md'][0]['lema'], float)






    # Can perform predictions with RSI & EMA
    def testPredictionWithRSIAndEMA(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)
        config['single_models'][0]['interpreter']['rsi'] = {'active': True}
        config['single_models'][0]['interpreter']['ema'] = {'active': True}

        # Initialize the model
        m = SingleModel(config)
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME)

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
        self.assertEqual(len(pred['md'][0]['pl']), m.arima['predictions'])
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)
        self.assertIsInstance(pred['md'][0]['rsi'], float)
        self.assertIsInstance(pred['md'][0]['sema'], float)
        self.assertIsInstance(pred['md'][0]['lema'], float)




    # Can predict without storing data in cache
    def testPredictWithoutCache(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = SingleModel(config)

        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, enable_cache=False)

        # Retrieve the prediction range
        first_ot, last_ct = Candlestick.get_current_prediction_range(config['single_models'][0]['lookback'], CURRENT_TIME)

        # Make sure the prediction was not stored
        cached_pred: Union[IPrediction, None] = get_prediction(config['id'], first_ot, last_ct)
        self.assertEqual(cached_pred, None)



    # Can predict storing data in cache
    def testPredictWithCache(self):
        # Init the config
        config = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = SingleModel(config)

        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, enable_cache=True)

        # Retrieve the prediction range
        first_ot, last_ct = Candlestick.get_current_prediction_range(config['single_models'][0]['lookback'], CURRENT_TIME)

        # Make sure the prediction was stored
        cached_pred: Union[IPrediction, None] = get_prediction(config['id'], first_ot, last_ct)
        self.assertFalse(cached_pred == None)
        self.assertDictEqual(pred, cached_pred)

        # Clean up the prediction
        delete_prediction(config['id'], first_ot, last_ct)
        cached_pred = get_prediction(config['id'], first_ot, last_ct)
        self.assertTrue(cached_pred == None)






# Test Execution
if __name__ == '__main__':
    unittest.main()