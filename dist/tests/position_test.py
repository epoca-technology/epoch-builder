import unittest
from typing import List
from pandas import Series, set_option
from modules.candlestick import Candlestick
from modules.utils import Utils
from modules.backtest import Position


# Make Panda's floats more readable
#set_option('display.float_format', lambda x: '%.2f' % x)


## Test Helpers


# Next Candlestick Retriever
def _get_next(candlestick: Series, change: float) -> Series:
    """Given a candlestick, it will fabricate the next given.

    Args:
        candlestick: Series
            The candlestick to base the new one on.
        change: float
            The change that will be applied to the prices.
    
    Returns:
        Series
    """
    c = candlestick.copy()
    c['ot'] = Utils.add_minutes(candlestick['ot'], 1)
    c['ct'] = Utils.add_minutes(candlestick['ot'], 2) - 1
    c['o'] = Utils.alter_number_by_percentage(candlestick['o'], change)
    c['h'] = Utils.alter_number_by_percentage(candlestick['h'], change)
    c['l'] = Utils.alter_number_by_percentage(candlestick['l'], change)
    c['c'] = Utils.alter_number_by_percentage(candlestick['c'], change)
    return c



# Start
START_CANDLESTICK: Series = Candlestick.DF.iloc[35888].copy()
START_CANDLESTICK['o'] = 40050.85
START_CANDLESTICK['h'] = 40068.52
START_CANDLESTICK['l'] = 40046.37
START_CANDLESTICK['c'] = 40052.18





## Test Class ##
class PositionTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    ## Positions ##


    # Can open and close a successful long position
    def testOpenAndCloseSuccessfulLongPosition(self):
        # Init the position
        p = Position(1, 1)

        # Init the test data
        c: List[Series] = [START_CANDLESTICK]
        pred = {"r": 1, "t": Utils.get_time()}

        # Open a long
        p.open_position(START_CANDLESTICK, pred)

        # Validate its integrity
        self.assertEqual(p.active['t'], 1)
        self.assertDictEqual(p.active['p'], pred)
        self.assertEqual(p.active['ot'], c[-1]['ot'])
        self.assertEqual(p.active['op'], c[-1]['o'])
        self.assertEqual(p.active['tpp'], Utils.alter_number_by_percentage(c[-1]['o'], 1))
        self.assertEqual(p.active['slp'], Utils.alter_number_by_percentage(c[-1]['o'], -1))

        # Add a new candlestick that does not touch the TP or the SL
        c.append(_get_next(c[-1], 0.1))

        # Checking the position should do nothing
        position_closed = p.check_position(c[-1])
        self.assertFalse(position_closed)
        self.assertEqual(len(p.positions), 0)
        self.assertIsInstance(p.active, dict)

        # Add a new candlestick that touches the TP
        c.append(_get_next(c[-1], 1))

        # Checking the position should close it
        position_closed = p.check_position(c[-1])
        self.assertTrue(position_closed)
        self.assertEqual(p.active, None)
        self.assertEqual(len(p.positions), 1)
        self.assertEqual(p.positions[0]['ct'], c[-1]['ct'])
        self.assertEqual(p.positions[0]['cp'], p.positions[0]['tpp'])
        self.assertTrue(p.positions[0]['o'])
        self.assertEqual(p.positions[0]['pts'], 0.9)

        # Make sure the points were added
        self.assertEqual(len(p.points), 2)
        self.assertEqual(p.points[-1], 0.9)

        # Check the counters
        self.assertEqual(p.successful_num, 1)
        self.assertEqual(p.long_num, 1)
        self.assertEqual(p.successful_long_num, 1)
        self.assertEqual(p.short_num, 0)
        self.assertEqual(p.successful_short_num, 0)





    # Can open and close an unsuccessful long position
    def testOpenAndCloseUnsuccessfulLongPosition(self):
        # Init the position
        p = Position(1, 1)

        # Init the test data
        c: List[Series] = [START_CANDLESTICK]
        pred = {"r": 1, "t": Utils.get_time()}

        # Open a long
        p.open_position(START_CANDLESTICK, pred)

        # Validate its integrity
        self.assertEqual(p.active['t'], 1)
        self.assertDictEqual(p.active['p'], pred)
        self.assertEqual(p.active['ot'], c[-1]['ot'])
        self.assertEqual(p.active['op'], c[-1]['o'])
        self.assertEqual(p.active['tpp'], Utils.alter_number_by_percentage(c[-1]['o'], 1))
        self.assertEqual(p.active['slp'], Utils.alter_number_by_percentage(c[-1]['o'], -1))

        # Add a new candlestick that touches the SL
        c.append(_get_next(c[-1], -1))

        # Checking the position should close it
        position_closed = p.check_position(c[-1])
        self.assertTrue(position_closed)
        self.assertEqual(p.active, None)
        self.assertEqual(len(p.positions), 1)
        self.assertEqual(p.positions[0]['ct'], c[-1]['ct'])
        self.assertEqual(p.positions[0]['cp'], p.positions[0]['slp'])
        self.assertFalse(p.positions[0]['o'])
        self.assertEqual(p.positions[0]['pts'], -1.1)

        # Make sure the points were added
        self.assertEqual(len(p.points), 2)
        self.assertEqual(p.points[-1], -1.1)

        # Check the counters
        self.assertEqual(p.successful_num, 0)
        self.assertEqual(p.long_num, 1)
        self.assertEqual(p.successful_long_num, 0)
        self.assertEqual(p.short_num, 0)
        self.assertEqual(p.successful_short_num, 0)




    # Can open and close a successful short position
    def testOpenAndCloseSuccessfulShortPosition(self):
        # Init the position
        p = Position(2.5, 2.5)

        # Init the test data
        c: List[Series] = [START_CANDLESTICK]
        pred = {"r": -1, "t": Utils.get_time()}

        # Open a short
        p.open_position(START_CANDLESTICK, pred)

        # Validate its integrity
        self.assertEqual(p.active['t'], -1)
        self.assertDictEqual(p.active['p'], pred)
        self.assertEqual(p.active['ot'], c[-1]['ot'])
        self.assertEqual(p.active['op'], c[-1]['o'])
        self.assertEqual(p.active['tpp'], Utils.alter_number_by_percentage(c[-1]['o'], -2.5))
        self.assertEqual(p.active['slp'], Utils.alter_number_by_percentage(c[-1]['o'], 2.5))

        # Add a new candlestick that does not touch the TP or the SL
        c.append(_get_next(c[-1], 0.89))

        # Checking the position should do nothing
        position_closed = p.check_position(c[-1])
        self.assertFalse(position_closed)
        self.assertEqual(len(p.positions), 0)
        self.assertIsInstance(p.active, dict)

        # Add a new candlestick that touches the TP
        c.append(_get_next(c[-1], -3.75))

        # Checking the position should close it
        position_closed = p.check_position(c[-1])
        self.assertTrue(position_closed)
        self.assertEqual(p.active, None)
        self.assertEqual(len(p.positions), 1)
        self.assertEqual(p.positions[0]['ct'], c[-1]['ct'])
        self.assertEqual(p.positions[0]['cp'], p.positions[0]['tpp'])
        self.assertTrue(p.positions[0]['o'])
        self.assertEqual(p.positions[0]['pts'], 2.4)

        # Make sure the points were added
        self.assertEqual(len(p.points), 2)
        self.assertEqual(p.points[-1], 2.4)


        # Check the counters
        self.assertEqual(p.successful_num, 1)
        self.assertEqual(p.long_num, 0)
        self.assertEqual(p.successful_long_num, 0)
        self.assertEqual(p.short_num, 1)
        self.assertEqual(p.successful_short_num, 1)



    # Can open and close an unsuccessful short position
    def testOpenAndCloseUnsuccessfulShortPosition(self):
        # Init the position
        p = Position(2.65, 3.89)

        # Init the test data
        c: List[Series] = [START_CANDLESTICK]
        pred = {"r": -1, "t": Utils.get_time()}

        # Open a short
        p.open_position(START_CANDLESTICK, pred)

        # Validate its integrity
        self.assertEqual(p.active['t'], -1)
        self.assertDictEqual(p.active['p'], pred)
        self.assertEqual(p.active['ot'], c[-1]['ot'])
        self.assertEqual(p.active['op'], c[-1]['o'])
        self.assertEqual(p.active['tpp'], Utils.alter_number_by_percentage(c[-1]['o'], -2.65))
        self.assertEqual(p.active['slp'], Utils.alter_number_by_percentage(c[-1]['o'], 3.89))

        # Add a new candlestick that touches the SL
        c.append(_get_next(c[-1], 6.65))

        # Checking the position should close it
        position_closed = p.check_position(c[-1])
        self.assertTrue(position_closed)
        self.assertEqual(p.active, None)
        self.assertEqual(len(p.positions), 1)
        self.assertEqual(p.positions[0]['ct'], c[-1]['ct'])
        self.assertEqual(p.positions[0]['cp'], p.positions[0]['slp'])
        self.assertFalse(p.positions[0]['o'])
        self.assertEqual(p.positions[0]['pts'], -3.99)

        # Make sure the points were added
        self.assertEqual(len(p.points), 2)
        self.assertEqual(p.points[-1], -3.99)

        # Check the counters
        self.assertEqual(p.successful_num, 0)
        self.assertEqual(p.long_num, 0)
        self.assertEqual(p.successful_long_num, 0)
        self.assertEqual(p.short_num, 1)
        self.assertEqual(p.successful_short_num, 0)





    # Can simulate a backtest by managing several successful and unsuccessful positions
    def testOpenAndCloseMultiplePositions(self):
        # Init the position & test data
        p = Position(1, 1)
        c: List[Series] = [START_CANDLESTICK]

        # Open a long position
        p.open_position(c[0], {'r': 1})

        # Add a few candlesticks that dont trigger a position close
        c.append(_get_next(c[-1], 0.36))
        c.append(_get_next(c[-1], -0.13))
        c.append(_get_next(c[-1], -0.08))
        c.append(_get_next(c[-1], -0.28))
        for index in range(1, len(c) - 1, 1):
            self.assertFalse(p.check_position(c[index]))

        # Close the position successfully
        c.append(_get_next(c[-1], 1.89))
        self.assertTrue(p.check_position(c[-1]))

        # Open another long position and close it unsuccessfully
        c.append(_get_next(c[-1], 0.54))
        p.open_position(c[-1], {'r': 1})
        c.append(_get_next(c[-1], -2.84))
        self.assertTrue(p.check_position(c[-1]))

        # Open a short position and close it successfully
        c.append(_get_next(c[-1], 2.11))
        p.open_position(c[-1], {'r': -1})
        c.append(_get_next(c[-1], -3.86))
        self.assertTrue(p.check_position(c[-1]))

        # Open another long position and close it successfully
        c.append(_get_next(c[-1], 1.76))
        p.open_position(c[-1], {'r': 1})
        c.append(_get_next(c[-1], 1.42))
        self.assertTrue(p.check_position(c[-1]))

        # Open a short position and close it unsuccessfully
        c.append(_get_next(c[-1], 0.65))
        p.open_position(c[-1], {'r': -1})
        c.append(_get_next(c[-1], 1.56))
        self.assertTrue(p.check_position(c[-1]))

        # There should be no active positions
        self.assertEqual(p.active, None)

        # Validate the integrity of the performance
        perf = p.get_performance()
        self.assertEqual(perf['points'], 0.5)
        self.assertEqual(perf['points_hist'][-1], 0.5)
        self.assertEqual(len(perf['points_hist']), 5 + 1)
        self.assertEqual(len(perf['positions']), 5)
        self.assertEqual(perf['long_num'], 3)
        self.assertEqual(perf['short_num'], 2)
        self.assertEqual(perf['long_acc'], 66.67)
        self.assertEqual(perf['short_acc'], 50)
        self.assertEqual(perf['general_acc'], 60)








    ## Exit Prices ##



    # Can get exit prices for both position types and any TP/SL Percentage
    def testGetExitPrices(self):
        # Test Price
        price = 40103.66

        # Init the position
        p = Position(1, 1)

        # Review the long exit params
        tp, sl = p._get_exit_prices(1, price)
        self.assertEqual(tp, Utils.alter_number_by_percentage(price, 1))
        self.assertEqual(sl, Utils.alter_number_by_percentage(price, -1))

        # Review the short exit params
        tp, sl = p._get_exit_prices(-1, price)
        self.assertEqual(tp, Utils.alter_number_by_percentage(price, -1))
        self.assertEqual(sl, Utils.alter_number_by_percentage(price, 1))


        # Init the position with different config
        p = Position(2.76, 3.11)

        # Review the long exit params
        tp, sl = p._get_exit_prices(1, price)
        self.assertEqual(tp, Utils.alter_number_by_percentage(price, 2.76))
        self.assertEqual(sl, Utils.alter_number_by_percentage(price, -3.11))

        # Review the short exit params
        tp, sl = p._get_exit_prices(-1, price)
        self.assertEqual(tp, Utils.alter_number_by_percentage(price, -2.76))
        self.assertEqual(sl, Utils.alter_number_by_percentage(price, 3.11))







    ## Points ##


    # Can update the points based on the position outcome
    def testUpdatePoints(self):
        # Init the position and simulate a backtest
        p = Position(1, 1)

        current = p._update_points(True)
        self.assertEqual(current, 0.9)
        self.assertEqual(len(p.points), 2)

        current = p._update_points(True)
        self.assertEqual(current, 1.8)
        self.assertEqual(len(p.points), 3)

        current = p._update_points(False)
        self.assertEqual(current, 0.7)
        self.assertEqual(len(p.points), 4)

        current = p._update_points(False)
        self.assertEqual(current, -0.4)
        self.assertEqual(len(p.points), 5)

        current = p._update_points(True)
        self.assertEqual(current, 0.5)
        self.assertEqual(len(p.points), 6)

        current = p._update_points(True)
        self.assertEqual(current, 1.4)
        self.assertEqual(len(p.points), 7)

        current = p._update_points(False)
        self.assertEqual(current, 0.3)
        self.assertEqual(len(p.points), 8)

        # Retrieve the performance and make sure the points list matches
        perf = p.get_performance()
        self.assertEqual(perf['points'], 0.3)
        self.assertEqual(len(perf['points_hist']), 8)






# Test Execution
if __name__ == '__main__':
    unittest.main()
