from typing import Union, List
import unittest
from modules._types import IProcessEarlyStoppingCheckpointConfig
from modules.database.Database import Database
from modules.process_early_stopping.ProcessEarlyStopping import ProcessEarlyStopping






## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TEST DATA

CHECKPOINTS: List[IProcessEarlyStoppingCheckpointConfig] = [
    { "required_longs": 1, "required_shorts": 1, "dataset_percent": 0.15 },
    { "required_longs": 3, "required_shorts": 3, "dataset_percent": 0.3 },
    { "required_longs": 7, "required_shorts": 7, "dataset_percent": 0.5 },
    { "required_longs": 10, "required_shorts": 10, "dataset_percent": 0.7 },
]





## Test Class ##
class ProcessEarlyStoppingTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    # Can evaluate a model that passes every checkpoint
    def testPassingModel(self):
        # Init the instance
        es = ProcessEarlyStopping(process_name="UnitTest", candlesticks_num=100, checkpoints=CHECKPOINTS, min_points=-35)

        # Run a check at 10%
        es_motive: Union[str, None] = es.check(points=5, current_index=10, longs_num=0, shorts_num=1)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], False)
        self.assertEqual(es.checkpoints[1]['passed'], False)
        self.assertEqual(es.checkpoints[2]['passed'], False)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Evaluate the first checkpoint
        es_motive = es.check(points=6.65, current_index=16, longs_num=2, shorts_num=2)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], False)
        self.assertEqual(es.checkpoints[2]['passed'], False)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Run a check at 20%
        es_motive = es.check(points=7.11, current_index=20, longs_num=2, shorts_num=3)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], False)
        self.assertEqual(es.checkpoints[2]['passed'], False)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Evaluate the second checkpoint
        es_motive = es.check(points=9.85, current_index=32, longs_num=3, shorts_num=4)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], False)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Run a check at 40%
        es_motive = es.check(points=10.4, current_index=40, longs_num=3, shorts_num=4)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], False)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Evaluate the third checkpoint
        es_motive = es.check(points=11.33, current_index=52, longs_num=10, shorts_num=11)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], True)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Run a check at 60%
        es_motive = es.check(points=12.67, current_index=60, longs_num=12, shorts_num=14)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], True)
        self.assertEqual(es.checkpoints[3]['passed'], False)

        # Evaluate the final checkpoint
        es_motive = es.check(points=14.33, current_index=73, longs_num=15, shorts_num=15)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], True)
        self.assertEqual(es.checkpoints[3]['passed'], True)

        # Complete the evaluation
        es_motive = es.check(points=20, current_index=100, longs_num=15, shorts_num=15)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], True)
        self.assertEqual(es.checkpoints[3]['passed'], True)







    # An evaluation must be stopped the moment a model hits -20 points
    def testUnacceptablePoints(self):
        # Init the instance
        es = ProcessEarlyStopping(process_name="UnitTest", candlesticks_num=100, checkpoints=CHECKPOINTS, min_points=-35)

        # Run a couple of checks
        es_motive = es.check(points=-3.3, current_index=5, longs_num=1, shorts_num=0)
        self.assertEqual(es_motive, None)
        es_motive = es.check(points=-6.9, current_index=10, longs_num=1, shorts_num=1)
        self.assertEqual(es_motive, None)

        # Should stop if -35 points are hit regardless of the checkpoints
        es_motive = es.check(points=-35, current_index=13, longs_num=1, shorts_num=2)
        self.assertIsInstance(es_motive, str)
        self.assertTrue("points" in es_motive)




    # An evaluation must be stopped if it fails the first checkpoint
    def testFailFirstCheckpoint(self):
        es = ProcessEarlyStopping(process_name="UnitTest", candlesticks_num=100, checkpoints=CHECKPOINTS)
        es_motive = es.check(current_index=16, longs_num=1, shorts_num=0)
        self.assertIsInstance(es_motive, str)
        self.assertEqual(es.checkpoints[0]['passed'], False)




    # An evaluation must be stopped if it fails the second checkpoint
    def testFailSecondCheckpoint(self):
        es = ProcessEarlyStopping(process_name="UnitTest", candlesticks_num=100, checkpoints=CHECKPOINTS)

        # Passes the first checkpoint
        es_motive = es.check(current_index=16, longs_num=1, shorts_num=1)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)

        # Fails on the second checkpoint
        es_motive = es.check(current_index=30, longs_num=3, shorts_num=2)
        self.assertIsInstance(es_motive, str)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], False)




    # An evaluation must be stopped if it fails the third checkpoint
    def testFailThirdCheckpoint(self):
        es = ProcessEarlyStopping(process_name="UnitTest", candlesticks_num=100, checkpoints=CHECKPOINTS, min_points=-35)

        # Passes the first checkpoint
        es_motive = es.check(points=-6.9, current_index=16, longs_num=1, shorts_num=1)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)

        # Passes the second checkpoint
        es_motive = es.check(points=-6.9, current_index=32, longs_num=4, shorts_num=5)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)

        # Fails on the third checkpoint
        es_motive = es.check(points=-3.3, current_index=52, longs_num=6, shorts_num=7)
        self.assertIsInstance(es_motive, str)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], False)




    # An evaluation must be stopped if it fails the fourth checkpoint
    def testFailFourthCheckpoint(self):
        es = ProcessEarlyStopping(process_name="UnitTest", candlesticks_num=100, checkpoints=CHECKPOINTS, min_points=-35)

        # Passes the first checkpoint
        es_motive = es.check(points=-6.9, current_index=16, longs_num=1, shorts_num=1)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)

        # Passes the second checkpoint
        es_motive = es.check(points=-6.9, current_index=32, longs_num=4, shorts_num=5)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)

        # Passes the third checkpoint
        es_motive = es.check(points=-6.9, current_index=56, longs_num=8, shorts_num=7)
        self.assertEqual(es_motive, None)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], True)

        # Fails on the fourth checkpoint
        es_motive = es.check(points=-3.3, current_index=75, longs_num=9, shorts_num=13)
        self.assertIsInstance(es_motive, str)
        self.assertEqual(es.checkpoints[0]['passed'], True)
        self.assertEqual(es.checkpoints[1]['passed'], True)
        self.assertEqual(es.checkpoints[2]['passed'], True)
        self.assertEqual(es.checkpoints[3]['passed'], False)







# Test Execution
if __name__ == '__main__':
    unittest.main()
