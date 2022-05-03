import unittest
from modules.candlestick import Candlestick
from modules.model import MultiModel, IModel, IPrediction, ISingleModelConfig


## Test Models ##
ID: str = 'MULTI_MODEL'
CONFIG_1: ISingleModelConfig = {
    'lookback': 150,
    'arima': { 'predictions': 10, 'p': 2, 'd': 1, 'q': 2 },
    'interpreter': {
        'long': 0.5, 
        'short': 0.5,
    }
}

CONFIG_2: ISingleModelConfig = {
    'lookback': 50,
    'arima': { 'predictions': 5, 'p': 2, 'd': 1, 'q': 4 },
    'interpreter': {
        'long': 0.75, 
        'short': 0.75,
    }
}

CONFIG_3: ISingleModelConfig = {
    'lookback': 100,
    'arima': { 'predictions': 10, 'p': 2, 'd': 1, 'q': 1 },
    'interpreter': {
        'long': 0.75, 
        'short': 0.75,
        'rsi': {'active': True}
    }
}

CONFIG_4: ISingleModelConfig = {
    'lookback': 200,
    'arima': { 'predictions': 10, 'p': 2, 'd': 1, 'q': 1 },
    'interpreter': {
        'long': 0.75, 
        'short': 0.75,
        'ema': {'active': True}
    }
}

CONFIG_5: ISingleModelConfig = {
    'lookback': 75,
    'arima': { 'predictions': 15, 'p': 2, 'd': 2, 'q': 1 },
    'interpreter': {
        'long': 1, 
        'short': 1,
        'rsi': {'active': True},
        'ema': {'active': True}
    }
}


## Current Timestamp ##
CURRENT_TIME: int = Candlestick.DF.iloc[35888]['ot']



## Test Class ##
class MultiModelTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass


    # Cannot initialize a MultiModel with an invalid consensus
    def testMustProvideValidConsensus(self):
        with self.assertRaises(ValueError):
            MultiModel({
                "id": ID,
                "consensus": 2,
                "single_models": [CONFIG_1, CONFIG_2, CONFIG_3, CONFIG_4]
            })

        with self.assertRaises(ValueError):
            MultiModel({
                "id": ID,
                "consensus": 3,
                "single_models": [CONFIG_1, CONFIG_2, CONFIG_3, CONFIG_4, CONFIG_5, CONFIG_2]
            })




    # Can Initialize a MultiModel with 5 Models in it and perform a prediction
    def testInititializeMultiModel(self):
        # Init the model
        mm: MultiModel = MultiModel({
            'id': ID,
            'single_models': [CONFIG_1, CONFIG_2, CONFIG_3, CONFIG_4, CONFIG_5]
        })

        # Validate the integrity
        s: IModel = mm.get_model()
        self.assertEqual(s['id'], ID)
        self.assertEqual(s['consensus'], 5)
        self.assertEqual(len(s['single_models']), 5)
        self.assertEqual(s['single_models'][0]['lookback'], CONFIG_1['lookback'])
        self.assertEqual(s['single_models'][1]['lookback'], CONFIG_2['lookback'])
        self.assertEqual(s['single_models'][2]['lookback'], CONFIG_3['lookback'])
        self.assertEqual(s['single_models'][3]['lookback'], CONFIG_4['lookback'])
        self.assertEqual(s['single_models'][4]['lookback'], CONFIG_5['lookback'])

        # Validate the max lookback
        self.assertEqual(mm.get_max_lookback(), 200)

        # Perform a prediction and validate its integrity
        pred: IPrediction = mm.predict(CURRENT_TIME)
        self.assertIsInstance(pred['r'], int)
        self.assertEqual(pred['t'], CURRENT_TIME)
        self.assertIsInstance(pred['md'], list)
        self.assertEqual(len(pred['md']), 5)
        



    # Can get the correct result based on the consensus and the results received from all the models
    def tesGetPredictionResult(self):
        # Init the model
        mm: MultiModel = MultiModel({
            'id': ID,
            'consensus': 3,
            'single_models': [CONFIG_1, CONFIG_2, CONFIG_3, CONFIG_4, CONFIG_5]
        })

        ## Validate the results' integrity ##

        # Longs
        self.assertEqual(mm._get_prediction_result([1, 1, 1, 1, 1]), 1)
        self.assertEqual(mm._get_prediction_result([1, 1, 1, 0, -1]), 1)
        self.assertEqual(mm._get_prediction_result([1, 1, -1, 0, 1]), 1)
        self.assertEqual(mm._get_prediction_result([1, 1, -1, 1, 1]), 1)
        self.assertEqual(mm._get_prediction_result([1, -1, -1, 1, 1]), 1)

        # Shorts
        self.assertEqual(mm._get_prediction_result([-1, -1, -1, -1, -1]), -1)
        self.assertEqual(mm._get_prediction_result([-1, -1, -1, 0, -1]), -1)
        self.assertEqual(mm._get_prediction_result([1, -1, -1, 0, -1]), -1)
        self.assertEqual(mm._get_prediction_result([-1, -1, 1, 0, -1]), -1)

        # Neutrals
        self.assertEqual(mm._get_prediction_result([0, 0, 0, 0, 0]), 0)
        self.assertEqual(mm._get_prediction_result([-1, 1, 0, 1, -1]), 0)
        self.assertEqual(mm._get_prediction_result([1, 0, -1, 1, -1]), 0)
        self.assertEqual(mm._get_prediction_result([1, 0, -1, -1, 0]), 0)



# Test Execution
if __name__ == '__main__':
    unittest.main()
