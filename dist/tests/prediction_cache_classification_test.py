from typing import List, TypedDict, Union
from unittest import TestCase, main
from copy import deepcopy
from modules._types import IPrediction
from modules.database.Database import Database
from modules.prediction_cache.ClassificationPredictionCache import ClassificationPredictionCache



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")


    


## Test Data ##

# Sample Type
class ISample(TypedDict):
    id: str
    first_ot: int
    last_ct: int
    min_probability: float
    pred: IPrediction

# Sample Data
SAMPLES: List[ISample] = [
    {
        "id": "C_UNIT_TEST",
        "first_ot": 1518719400004,
        "last_ct": 1518825599997,
        "min_probability": 0.65,
        "pred": {'r': 1,'t': 1518825599997,'md': [{'d': 'long','f': [1, 0, -1, 1, 1, 0.154855, -0.085694, 0.1455541], 'up': 0.645842185, 'dp': 0.36472156}]}
    },
    {
        "id": "C_UNIT_TEST_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1518888600007,
        "last_ct": 1519091999994,
        "min_probability": 0.78,
        "pred": {'r': 1,'t': 1519091999994,'md': [{'d': 'long','f': [-1, -1, -1, 0, 1, -0.814855, -0.685694, 0.0015541], 'up': 0.415459312, 'dp': 0.69121420546}]}
    },
    {
        "id": "C_CLSTM_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "min_probability": 0.55,
        "pred": {'r': 0,'t': 1620536399999,'md': [{'d': 'neutral','f': [0, 1, 0, 0, 0, -0.006855, -0.004194, 0.0855541], 'up': 0.51565451, 'dp': 0.496515447}]}
    },
    {
        "id": "C_CNN_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "min_probability": 0.6,
        "pred": {'r': 0,'t': 1620536399999,'md': [{'d': 'neutral'}]}
    }
]








## Test Helpers ##


def _get_cache(sample: ISample) -> ClassificationPredictionCache:
    """Returns a cache instance based on a sample.
    """
    return ClassificationPredictionCache(model_id=sample["id"], min_probability=sample["min_probability"])




def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for sample in SAMPLES:
        cache = _get_cache(sample)
        cache.delete(sample["first_ot"], sample["last_ct"])






# Test Class
class ClassificationPredictionCacheTestCase(TestCase):
    # Before Tests
    def setUp(self):
        _delete_tests()

    # After Tests
    def tearDown(self):
        _delete_tests()





    # Can save, retrieve and delete a series of predictions.
    def testClassificationCachingFlow(self):
        # Save the samples and validate their integrities
        for sample in SAMPLES:
            # Init the cache
            cache: ClassificationPredictionCache = _get_cache(sample)

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
            cache: ClassificationPredictionCache = _get_cache(sample)

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
        sample['min_probability'] = 0.50

        # Init the cache
        cache: ClassificationPredictionCache = _get_cache(sample)

        # Store the sample
        cache.save(sample['first_ot'], sample['last_ct'], sample['pred'])

        # Make sure the sample can be found
        sample['min_probability'] = 0.5
        cache_2: ClassificationPredictionCache = _get_cache(sample)
        pred: Union[IPrediction, None] = cache_2.get(sample['first_ot'], sample['last_ct'])
        self.assertEqual(pred, sample['pred'])

        # Make sure the sample can be found
        sample['min_probability'] = 0.50
        cache_3: ClassificationPredictionCache = _get_cache(sample)
        pred = cache_3.get(sample['first_ot'], sample['last_ct'])

        # Delete the sample
        cache_3.delete(sample['first_ot'], sample['last_ct'])

        # Make sure the sample is gone
        pred = cache_3.get(sample['first_ot'], sample['last_ct'])
        self.assertEqual(pred, None)



# Test Execution
if __name__ == '__main__':
    main()