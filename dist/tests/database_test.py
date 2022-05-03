import unittest
from typing import List, Union
from modules.model import IPrediction
from modules.database import get_prediction, save_prediction, delete_prediction


## Test Data ##
ids: List[str] = ['A111', 'A751', 'A449']
first_ot: List[int] = [1518719400004, 1518888600007, 1519209000002]
last_ct: List[int] = [1518825599997, 1519091999994, 1519361999991]
test_pred_1: IPrediction = {'r': 1,'t': 1518825599997,'md': [{'d': 'long','pl': [11545.65, 11611.85, 11635.41, 11666.99, 11785.14]}]}
test_pred_2: IPrediction = {'r': -1,'t': 1518825794597,'md': [{'d': 'short','pl': [11545.65, 11525.85, 11510.63, 11485.36, 11320.55]}]}
test_pred_3: IPrediction = {'r': 0,'t': 1518825691463,'md': [{'d': 'neutral','pl': [11545.65, 11541.36, 11542.84, 11547.65, 11529.85]}]}
prediction: List[IPrediction] = [test_pred_1, test_pred_2, test_pred_3]


## Test Helpers
def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for i, id in enumerate(ids):
        delete_prediction(id, first_ot[i], last_ct[i])






# Test Class
class DatabaseTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        _delete_tests()

    # After Tests
    def tearDown(self):
        _delete_tests()





    # Can save, retrieve and delete a series of arima prediction.
    def testSaveArimaPredictions(self):
        # Saving
        for i, id in enumerate(ids):
            # Save a prediction
            save_prediction(id, first_ot[i], last_ct[i], prediction[i])
            
            # Retrieve it and validate its integrity
            pred: IPrediction = get_prediction(id, first_ot[i], last_ct[i])
            self.assertDictEqual(pred, prediction[i])

        # Deleting
        for i, id in enumerate(ids):
            # Delete the prediction and make sure it is gone
            delete_prediction(id, first_ot[i], last_ct[i])
            pred: Union[IPrediction, None] = get_prediction(id, first_ot[i], last_ct[i])
            self.assertEqual(pred, None)




# Test Execution
if __name__ == '__main__':
    unittest.main()