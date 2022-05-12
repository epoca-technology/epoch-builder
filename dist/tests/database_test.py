import unittest
from typing import List, Union
from modules.model import IPrediction
from modules.database import get_pred, save_pred, delete_pred


## Test Data ##
long_id_1: str = 'R_DHS_5122561286432_relu_LB50_P5_LR001_ADAM_MAE_TP15SL1'
long_id_2: str = 'R_ARLHST_5122561286432_relu_tahn_relu_LB300_P10_LR0001_ADAM_MAE_TP15SL1'
ids: List[str] = ['A111', 'A751', 'A449', long_id_1, long_id_2]
first_ot: List[int] = [1518719400004, 1518888600007, 1519209000002, 1509174615789, 1509264004278]
last_ct: List[int] = [1518825599997, 1519091999994, 1519361999991, 1509218123144, 1509325132641]
test_pred_1: IPrediction = {'r': 1,'t': 1518825599997,'md': [{'d': 'long','pl': [11545.65, 11611.85, 11635.41, 11666.99, 11785.14]}]}
test_pred_2: IPrediction = {'r': -1,'t': 1518825794597,'md': [{'d': 'short','pl': [11545.65, 11525.85, 11510.63, 11485.36, 11320.55]}]}
test_pred_3: IPrediction = {'r': 0,'t': 1518825691463,'md': [{'d': 'neutral','pl': [11545.65, 11541.36, 11542.84, 11547.65, 11529.85]}]}
test_pred_4: IPrediction = {'r': 1,'t': 1518825691463,'md': [{'d': 'long','npl': [0.0651, 0.0655, 0.0687, 0.0699, 0.07115]}]}
test_pred_5: IPrediction = {'r': -1,'t': 1518825691463,'md': [{'d': 'short','npl': [0.1854, 0.184325, 0.182551, 0.186451, 0.181366, 0.18023, 0.180012, 0.17985, 0.1799558]}]}
prediction: List[IPrediction] = [test_pred_1, test_pred_2, test_pred_3, test_pred_4, test_pred_5]


## Test Helpers
def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for i, id in enumerate(ids):
        delete_pred(id, first_ot[i], last_ct[i])






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
            save_pred(id, first_ot[i], last_ct[i], prediction[i])
            
            # Retrieve it and validate its integrity
            pred: IPrediction = get_pred(id, first_ot[i], last_ct[i])
            self.assertDictEqual(pred, prediction[i])

        # Deleting
        for i, id in enumerate(ids):
            # Delete the prediction and make sure it is gone
            delete_pred(id, first_ot[i], last_ct[i])
            pred: Union[IPrediction, None] = get_pred(id, first_ot[i], last_ct[i])
            self.assertEqual(pred, None)




# Test Execution
if __name__ == '__main__':
    unittest.main()