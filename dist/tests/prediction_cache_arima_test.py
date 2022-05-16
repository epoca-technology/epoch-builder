from typing import List, TypedDict, Union
from unittest import TestCase, main
from copy import deepcopy
from modules.model import IPrediction
from modules.prediction_cache import get_arima_pred, save_arima_pred, delete_arima_pred



## Test Data ##

# Sample Type
class ISample(TypedDict):
    id: str
    first_ot: int
    last_ct: int
    predictions: int
    long: float
    short: float
    pred: IPrediction

# Sample Data
SAMPLES: List[ISample] = [
    {
        "id": "A111",
        "first_ot": 1518719400004,
        "last_ct": 1518825599997,
        "predictions": 10,
        "long": 0.05,
        "short": 0.05,
        "pred": {'r': 1,'t': 1518825599997,'md': [{'d': 'long','pl': [11545.65, 11611.85, 11635.41, 11666.99, 11785.14]}]}
    },
    {
        "id": "A751",
        "first_ot": 1518888600007,
        "last_ct": 1519091999994,
        "predictions": 15,
        "long": 0.18,
        "short": 0.14,
        "pred": {'r': -1,'t': 1518825794597,'md': [{'d': 'short','pl': [11545.65, 11525.85, 11510.63, 11485.36, 11320.55, 11318.63, 11654,11, 10854.01, 10594, 10574.54]}]}
    },
    {
        "id": "A449",
        "first_ot": 1519209000002,
        "last_ct": 1519361999991,
        "predictions": 5,
        "long": 5.90,
        "short": 10.14,
        "pred": {'r': 0,'t': 1518825691463,'md': [{'d': 'neutral','pl': [11545.65, 11541.36, 11542.84, 11547.65, 11529.85]}]}
    },
    {
        "id": "A8654518",
        "first_ot": 1509174615789,
        "last_ct": 1509218123144,
        "predictions": 10,
        "long": 1.10,
        "short": 1.55,
        "pred": {'r': 1,'t': 1518848914363,'md': [{'d': 'long'}]}
    },
    {
        "id": "A2124553",
        "first_ot": 1509264004278,
        "last_ct": 1509325132641,
        "predictions": 25,
        "long": 3.85,
        "short": 1.90,
        "pred": {'r': -1,'t': 1567411085379,'md': [{'d': 'short'}]}
    },
    {
        "id": "A6854443",
        "first_ot": 1509267841189,
        "last_ct": 1515463974141,
        "predictions": 10,
        "long": 1,
        "short": 2,
        "pred": {'r': 0,'t': 1567411677759,'md': [{'d': 'neutral'}]}
    },
    {
        "id": "A7884559",
        "first_ot": 1509999885489,
        "last_ct": 15159994145641,
        "predictions": 5,
        "long": 1.0,
        "short": 2.00,
        "pred": {'r': 0,'t': 1567411666659,'md': [{'d': 'neutral'}]}
    },
]







## Test Helpers ##
def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for s in SAMPLES:
        delete_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])






# Test Class
class ArimaPredictionCacheTestCase(TestCase):
    # Before Tests
    def setUp(self):
        _delete_tests()

    # After Tests
    def tearDown(self):
        _delete_tests()





    # Can save, retrieve and delete a series of arima predictions.
    def testArimaCachingFlow(self):
        # Save the samples and validate their integrities
        for s in SAMPLES:
            # Make sure the sample does not exist
            pred: Union[IPrediction, None] = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])
            self.assertEqual(pred, None)

            # Store the sample
            save_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'], s['pred'])

            # Retrieve it and validate its integrity
            pred = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])
            self.assertDictEqual(pred, s['pred'])

        # Iterate over each sample, delete it and verify it has been deleted
        for s in SAMPLES:
            # The sample should exist
            pred: Union[IPrediction, None] = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])
            self.assertDictEqual(pred, s['pred'])

            # Delete the sample
            delete_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])

            # Make sure the sample is gone
            pred = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])
            self.assertEqual(pred, None)





    # Can save a prediction with inexact interpreter configs but still make the match
    def testInterpreterConfigMatching(self):
        # Init the sample
        s: ISample = deepcopy(SAMPLES[0])
        s['long'] = 1.50
        s['short'] = 2.00

        # Store the sample
        save_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'], s['pred'])

        # Make sure the sample can be found
        pred: Union[IPrediction, None] = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], 1.5, 2)
        self.assertEqual(pred, s['pred'])

        # Make sure the sample can be found
        pred = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], 1.50, 2.0)
        self.assertEqual(pred, s['pred'])

        # Make sure the sample can be found
        pred = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], 1.50, 2.00)
        self.assertEqual(pred, s['pred'])

        # Delete the sample
        delete_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])

        # Make sure the sample is gone
        pred = get_arima_pred(s['id'], s['first_ot'], s['last_ct'], s['predictions'], s['long'], s['short'])
        self.assertEqual(pred, None)



# Test Execution
if __name__ == '__main__':
    main()