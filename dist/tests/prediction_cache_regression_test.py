from typing import List, TypedDict, Union
from unittest import TestCase, main
from copy import deepcopy
from modules.types import IPrediction
from modules.database.Database import Database
from modules.prediction_cache.RegressionPredictionCache import RegressionPredictionCache



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")


    


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
    {
        "id": "R_UNIT_TEST",
        "first_ot": 1614504600000,
        "last_ct": 1614589199999,
        "predictions": 30,
        "long": 1.5,
        "short": 2.11,
        "pred": {'r': 1,'t': 1614504600000,'md': [{'d': 'long'}]}
    },
    {
        "id": "R_DNN_S2_DO_YGIYAT63GZZ5HARUVFSX",
        "first_ot": 1614652200000,
        "last_ct": 1614736799999,
        "predictions": 30,
        "long": 1.89,
        "short": 3.64,
        "pred": {'r': -1,'t': 1614652200000,'md': [{'d': 'short'}]}
    },
    {
        "id": "R_CLSTM_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "predictions": 30,
        "long": 2,
        "short": 5.79,
        "pred": {'r': 0,'t': 1620480600000,'md': [{'d': 'neutral'}]}
    }
]






## Test Helpers ##



def _get_cache(sample: ISample) -> RegressionPredictionCache:
    """Returns a cache instance based on a sample.
    """
    return RegressionPredictionCache(
        model_id=sample["id"],
        predictions=sample["predictions"],
        interpreter_long=sample["long"],
        interpreter_short=sample["short"],
    )



def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for sample in SAMPLES:
        cache = _get_cache(sample)
        cache.delete(sample["first_ot"], sample["last_ct"])






# Test Class
class RegressionPredictionCacheTestCase(TestCase):
    # Before Tests
    def setUp(self):
        _delete_tests()

    # After Tests
    def tearDown(self):
        _delete_tests()





    # Can save, retrieve and delete a series of predictions.
    def testRegressionCachingFlow(self):
        # Save the samples and validate their integrities
        for sample in SAMPLES:
            # Init the cache
            cache: RegressionPredictionCache = _get_cache(sample)

            # Make sure the sample does not exist
            pred: Union[IPrediction, None] = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(pred, None)

            # Store the sample
            cache.save(sample['first_ot'], sample['last_ct'], sample['pred'])

            # Retrieve it and validate its integrity
            pred = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertDictEqual(pred, sample['pred'])

        # Iterate over each sample, delete it and verify it has been deleted
        for sample in SAMPLES:
            # Init the cache
            cache: RegressionPredictionCache = _get_cache(sample)

            # The sample should exist
            pred: Union[IPrediction, None] = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertDictEqual(pred, sample['pred'])

            # Delete the sample
            cache.delete(sample['first_ot'], sample['last_ct'])

            # Make sure the sample is gone
            pred = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(pred, None)





    # Can save a prediction with inexact interpreter configs but still make the match
    def testInterpreterConfigMatching(self):
        # Init the sample
        sample: ISample = deepcopy(SAMPLES[0])
        sample['long'] = 1.50
        sample['short'] = 2.00

        # Init the cache
        cache: RegressionPredictionCache = _get_cache(sample)

        # Store the sample
        cache.save(sample['first_ot'], sample['last_ct'], sample['pred'])

        # Make sure the sample can be found
        sample['long'] = 1.5
        sample['short'] = 2
        cache_2: RegressionPredictionCache = _get_cache(sample)
        pred: Union[IPrediction, None] = cache_2.get(sample['first_ot'], sample['last_ct'])
        self.assertEqual(pred, sample['pred'])

        # Make sure the sample can be found
        sample['long'] = 1.50
        sample['short'] = 2.0
        cache_3: RegressionPredictionCache = _get_cache(sample)
        pred = cache_3.get(sample['first_ot'], sample['last_ct'])
        self.assertEqual(pred, sample['pred'])

        # Make sure the sample can be found
        sample['long'] = 1.50
        sample['short'] = 2.00
        cache_4: RegressionPredictionCache = _get_cache(sample)
        pred = cache_4.get(sample['first_ot'], sample['last_ct'])
        self.assertEqual(pred, sample['pred'])

        # Delete the sample
        cache_4.delete(sample['first_ot'], sample['last_ct'])

        # Make sure the sample is gone
        pred = cache_4.get(sample['first_ot'], sample['last_ct'])
        self.assertEqual(pred, None)



# Test Execution
if __name__ == '__main__':
    main()