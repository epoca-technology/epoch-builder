from typing import List, TypedDict, Union
from unittest import TestCase, main
from modules._types import IPrediction
from modules.database.Database import Database
from modules.prediction_cache.PredictionCache import PredictionCache



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")


    


## Test Data ##

# Sample Type
class ISample(TypedDict):
    id: str
    first_ot: int
    last_ct: int
    pred: IPrediction

# Sample Data
SAMPLES: List[ISample] = [
    {
        "id": "KR_UNIT_TEST",
        "first_ot": 1614504600000,
        "last_ct": 1614589199999,
        "pred": {'r': 1,'t': 1614504600000,'md': [{'d': 'long', 'pl': [0.470684114904175, 0.48970961570739746, 0.4899211525917053, 0.4895859956741333, 0.4902108311653137, 0.4903203547000885, 0.48996052145957947, 0.48973268270492554, 0.4894953966140747, 0.4905127286911011, 0.49087634682655334, 0.4910549819469452, 0.49125850200653076, 0.49153727293014526, 0.4920978844165802, 0.49207258224487305, 0.4915007948875427, 0.4912847876548767, 0.4900345206260681, 0.4913979768753052, 0.4916602373123169, 0.4913593530654907, 0.49184101819992065, 0.49248355627059937, 0.4916429817676544, 0.49207931756973267, 0.4939451813697815, 0.49415746331214905, 0.4942833185195923, 0.4945600926876068, 0.495000422000885]}]}
    },
    {
        "id": "KR_DNN_S2_DO_YGIYAT63GZZ5HARUVFSX",
        "first_ot": 1614652200000,
        "last_ct": 1614736799999,
        "pred": {'r': -1,'t': 1614652200000,'md': [{'d': 'short', 'pl': [0.5002286970859469, 0.5160702466964722, 0.5158073902130127, 0.5151525735855103, 0.5146346092224121, 0.5144381523132324, 0.51382976770401, 0.5134482979774475, 0.5133419036865234, 0.5130729079246521, 0.5134098529815674, 0.5129741430282593, 0.5118387937545776, 0.5119805932044983, 0.5126540064811707, 0.5122087001800537, 0.5117754340171814, 0.5122472643852234, 0.5130888819694519, 0.5137238502502441, 0.5138534307479858, 0.5140414237976074, 0.5151236653327942, 0.5146328806877136, 0.5144575238227844, 0.5153934955596924, 0.5169110894203186, 0.5171418190002441, 0.5184459090232849, 0.5194806456565857, 0.5204160809516907]}]}
    },
    {
        "id": "KR_CLSTM_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "pred": {'r': 0,'t': 1620480600000,'md': [{'d': 'neutral'}]}
    },
    {
        "id": "KC_UNIT_TEST",
        "first_ot": 1518719400004,
        "last_ct": 1518825599997,
        "pred": {'r': 1,'t': 1518825599997,'md': [{'d': 'long','f': [1, 0, -1, 1, 1, 0.154855, -0.085694, 0.1455541], 'up': 0.645842185, 'dp': 0.36472156}]}
    },
    {
        "id": "KC_UNIT_TEST_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1518888600007,
        "last_ct": 1519091999994,
        "pred": {'r': -1,'t': 1519091999994,'md': [{'d': 'long','f': [-1, -1, -1, 0, 1, -0.814855, -0.685694, 0.0015541], 'up': 0.415459312, 'dp': 0.69121420546}]}
    },
    {
        "id": "KC_CLSTM_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "pred": {'r': 0,'t': 1620536399999,'md': [{'d': 'neutral','f': [0, 1, 0, 0, 0, -0.006855, -0.004194, 0.0855541], 'up': 0.51565451, 'dp': 0.496515447}]}
    },
    {
        "id": "KC_CNN_S4_MP_DO_5b0133d4-309b-44ae-93e9-c952e76dec34",
        "first_ot": 1620480600000,
        "last_ct": 1620536399999,
        "pred": {'r': 0,'t': 1620536399999,'md': [{'d': 'neutral'}]}
    }
]








## Test Helpers ##




def _delete_tests() -> None:
    """Deletes all the tests in order to ensure a fresh start every time.
    """
    for sample in SAMPLES:
        cache = PredictionCache(sample['id'])
        cache.delete(sample["first_ot"], sample["last_ct"])






# Test Class
class PredictionCacheTestCase(TestCase):
    # Before Tests
    def setUp(self):
        _delete_tests()

    # After Tests
    def tearDown(self):
        _delete_tests()





    # Can save, retrieve and delete a series of predictions.
    def testPredictionCacheFlow(self):
        # Save the samples and validate their integrities
        for sample in SAMPLES:
            # Init the cache
            cache: PredictionCache = PredictionCache(sample['id'])

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
            cache: PredictionCache = PredictionCache(sample['id'])

            # The sample should exist
            pred: Union[IPrediction, None] = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertDictEqual(pred, sample['pred'])

            # Delete the sample
            cache.delete(sample['first_ot'], sample['last_ct'])

            # Make sure the sample is gone
            pred = cache.get(sample['first_ot'], sample['last_ct'])
            self.assertEqual(pred, None)







# Test Execution
if __name__ == '__main__':
    main()