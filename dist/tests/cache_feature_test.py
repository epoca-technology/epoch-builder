from typing import List, TypedDict, Union
from unittest import TestCase, main
from modules.database.Database import Database
from modules.prediction_cache.FeatureCache import FeatureCache



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")


    


## Test Data ##

# Sample Type
class ISample(TypedDict):
    id: str
    first_ot: int
    last_ct: int
    feature: float

# Sample Data
SAMPLES: List[ISample] = [
    {
        "id": "KR_UNIT_TEST",
        "first_ot": 1614504600000,
        "last_ct": 1614589199999,
        "feature": 0.512466
    },
    {
        "id": "KR_DNN_S2_DO_YGIYAT63GZZ5HARUVFSX",
        "first_ot": 1614652200000,
        "last_ct": 1614736799999,
        "feature": -0.845654
    },
    {
        "id": "KR_CLSTM_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "feature": -1
    },
    {
        "id": "KR_CLSTM_S4_DO_5b0133d4-309b-44ae-93e9-c952e76dec79",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "feature": 1
    },
    {
        "id": "KR_LSTM_S4_5b0133d4-309b-44ae-93e9-c952e76dec79",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "feature": 0
    }
]








## Test Helpers ##




def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for sample in SAMPLES:
        cache = FeatureCache(sample['id'])
        cache.delete(sample["first_ot"], sample["last_ct"])






# Test Class
class FeatureCacheTestCase(TestCase):
    # Before Tests
    def setUp(self):
        _delete_tests()

    # After Tests
    def tearDown(self):
        _delete_tests()





    # Can save, retrieve and delete a series of features.
    def testFeatureCacheFlow(self):
        # Save the samples and validate their integrities
        for sample in SAMPLES:
            # Init the cache
            cache: FeatureCache = FeatureCache(sample['id'])

            # Make sure the sample does not exist
            feature: Union[float, None] = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(feature, None)

            # Store the sample
            cache.save(sample['first_ot'], sample['last_ct'], sample['feature'])

            # Retrieve it and validate its integrity
            feature = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(feature, sample['feature'])

        # Iterate over each sample, delete it and verify it has been deleted
        for sample in SAMPLES:
            # Init the cache
            cache: FeatureCache = FeatureCache(sample["id"])

            # The sample should exist
            feature: Union[float, None] = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(feature, sample['feature'])

            # Delete the sample
            cache.delete(sample['first_ot'], sample['last_ct'])

            # Make sure the sample is gone
            feature = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(feature, None)







# Test Execution
if __name__ == '__main__':
    main()