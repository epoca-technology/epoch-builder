import unittest
from typing import Union
from copy import deepcopy
from pandas import DataFrame
from modules.types import IModel, IPrediction
from modules.database.Database import Database
from modules.candlestick.Candlestick import Candlestick
from modules.prediction_cache.ArimaPredictionCache import get_arima_pred, delete_arima_pred
from modules.model.ArimaModel import ArimaModel


## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")



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

        # Make sure the instance is recognized
        self.assertIsInstance(model, ArimaModel)
        self.assertEqual(type(model).__name__, "ArimaModel")

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
        cached_pred: Union[IPrediction, None] = get_arima_pred(
            m.id, 
            first_ot, 
            last_ct,
            m.predictions,
            m.interpreter.long,
            m.interpreter.short
        )
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
        cached_pred: Union[IPrediction, None] = get_arima_pred(
            m.id, 
            first_ot, 
            last_ct,
            m.predictions,
            m.interpreter.long,
            m.interpreter.short
        )
        self.assertFalse(cached_pred == None)
        self.assertDictEqual(pred, cached_pred)
        self.assertEqual(cached_pred['md'][0].get('pl'), None)

        # Clean up the prediction
        delete_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
        cached_pred = get_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
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
        cached_pred: Union[IPrediction, None] = get_arima_pred(
            m.id, 
            first_ot, 
            last_ct,
            m.predictions,
            m.interpreter.long,
            m.interpreter.short
        )
        self.assertFalse(cached_pred == None)
        self.assertDictEqual(pred, cached_pred)
        self.assertEqual(cached_pred['md'][0].get('pl'), None)

        # Clean up the prediction
        delete_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
        cached_pred = get_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
        self.assertTrue(cached_pred == None)




    # Can perform predictions with a provided matching lookback_df
    def testBasicPredictionWithLookbackDF(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)

        # Retrieve the DF
        df: DataFrame = Candlestick.get_lookback_df(150, CURRENT_TIME)
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, lookback_df=df, enable_cache=False)

        # Validate the integrity of the result
        self.assertIsInstance(pred['r'], int)
        self.assertTrue(pred['r'] == -1 or pred['r'] == 0 or pred['r'] == 1)
        self.assertEqual(pred['t'], CURRENT_TIME)
        self.assertIsInstance(pred['md'], list)
        self.assertEqual(len(pred['md']), 1)
        self.assertIsInstance(pred['md'][0], dict)
        self.assertIsInstance(pred['md'][0]['pl'], list)
        self.assertEqual(len(pred['md'][0]['pl']), m.predictions)
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)




    # Can perform predictions with a provided lookback_df that is bigger than the model's
    def testBasicPredictionWithDifferentLookbackDF(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)

        # Retrieve the DF
        df: DataFrame = Candlestick.get_lookback_df(300, CURRENT_TIME)
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, lookback_df=df, enable_cache=False)

        # Validate the integrity of the result
        self.assertIsInstance(pred['r'], int)
        self.assertTrue(pred['r'] == -1 or pred['r'] == 0 or pred['r'] == 1)
        self.assertEqual(pred['t'], CURRENT_TIME)
        self.assertIsInstance(pred['md'], list)
        self.assertEqual(len(pred['md']), 1)
        self.assertIsInstance(pred['md'][0], dict)
        self.assertIsInstance(pred['md'][0]['pl'], list)
        self.assertEqual(len(pred['md'][0]['pl']), m.predictions)
        self.assertIsInstance(pred['md'][0]['d'], str)
        self.assertTrue(len(pred['md'][0]['d']) > 0)




    # Can perform predictions with a provided matching lookback_df and cache
    def testBasicPredictionWithLookbackDFAndCache(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)

        # Retrieve the DF
        df: DataFrame = Candlestick.get_lookback_df(150, CURRENT_TIME)
        first_ot: int = int(df.iloc[0]["ot"])
        last_ct: int = int(df.iloc[-1]["ct"])
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, lookback_df=df, enable_cache=True)

        # Validate the integrity of the result
        self.assertIsInstance(pred['r'], int)
        self.assertTrue(pred['r'] == -1 or pred['r'] == 0 or pred['r'] == 1)
        self.assertEqual(pred['t'], CURRENT_TIME)
        self.assertIsInstance(pred['md'], list)

        # Clean up the prediction
        delete_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
        cached_pred = get_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
        self.assertTrue(cached_pred == None)





    # Can perform predictions with a provided different lookback_df and cache
    def testBasicPredictionWithDifferentLookbackDFAndCache(self):
        # Init the config
        config: IModel = deepcopy(BASIC_CONFIG)

        # Initialize the model
        m = ArimaModel(config)

        # Retrieve the DF
        df: DataFrame = Candlestick.get_lookback_df(300, CURRENT_TIME)
        sliced_df: DataFrame = df.iloc[-150:]
        first_ot: int = int(sliced_df.iloc[0]["ot"])
        last_ct: int = int(sliced_df.iloc[-1]["ct"])
        
        # Perform a prediction
        pred: IPrediction = m.predict(CURRENT_TIME, lookback_df=df, enable_cache=True)

        # Validate the integrity of the result
        self.assertIsInstance(pred['r'], int)
        self.assertTrue(pred['r'] == -1 or pred['r'] == 0 or pred['r'] == 1)
        self.assertEqual(pred['t'], CURRENT_TIME)
        self.assertIsInstance(pred['md'], list)

        # Clean up the prediction
        delete_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
        cached_pred = get_arima_pred(m.id, first_ot, last_ct, m.predictions, m.interpreter.long, m.interpreter.short)
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