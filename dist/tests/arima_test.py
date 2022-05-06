from typing import List
import unittest
from pandas import Series
from modules.arima import Arima




# Test Class
class ArimaTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    # Can perform an Arima Prediction
    def testArimaPrediction(self):
        # Init the instance
        arima: Arima = Arima({'p': 1, 'd': 2, 'q': 3}, 5)

        # Make sure it was initialized correctly
        self.assertEqual(arima.config['p'], 1)
        self.assertEqual(arima.config['d'], 2)
        self.assertEqual(arima.config['q'], 3)
        self.assertEqual(arima.config['P'], 0)
        self.assertEqual(arima.config['D'], 0)
        self.assertEqual(arima.config['Q'], 0)
        self.assertEqual(arima.config['m'], 0)
        self.assertEqual(arima.predictions, 5)

        # Perform a prediction
        preds: List[float] = arima.predict(Series([100.13, 101, 102.22, 102.5, 103.8, 105.55, 104.33]))
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 5)






    # Can perform a Sarima Prediction
    def testSarimaPrediction(self):
        # Init the instance
        arima: Arima = Arima({'p': 2, 'd': 1, 'q': 3, 'P': 1, 'D': 2, 'Q': 4, 'm': 4}, 10)

        # Make sure it was initialized correctly
        self.assertEqual(arima.config['p'], 2)
        self.assertEqual(arima.config['d'], 1)
        self.assertEqual(arima.config['q'], 3)
        self.assertEqual(arima.config['P'], 1)
        self.assertEqual(arima.config['D'], 2)
        self.assertEqual(arima.config['Q'], 4)
        self.assertEqual(arima.config['m'], 4)
        self.assertEqual(arima.predictions, 10)

        # Perform a prediction
        preds: List[float] = arima.predict(Series([100.13, 101, 102.22, 102.5, 103.8, 105.55, 104.33]))
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 10)








# Test Execution
if __name__ == '__main__':
    unittest.main()