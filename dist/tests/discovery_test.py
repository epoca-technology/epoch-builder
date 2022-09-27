from typing import List
from unittest import TestCase, main
from numpy import mean, median
from modules._types import IDiscovery
from modules.utils.Utils import Utils
from modules.discovery.Discovery import Discovery





## Helpers ##





## Test Class ##
class DiscoveryTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass





    # Initializes an instance performs simulations and validates the integrity of the data
    def testDiscoveryProcess(self):
        # Initialize the instance
        disc = Discovery(reward=1, penalty=1.3)
        current_points: float = 0

        # Simulate a neutral prediction
        self.assertEqual(disc.neutral_num, 0)
        disc.neutral_pred()
        self.assertEqual(disc.neutral_num, 1)
        disc.neutral_pred()
        self.assertEqual(disc.neutral_num, 2)

        # Simulate a neutral outcome
        self.assertEqual(disc.neutral_outcome_num, 0)
        disc.neutral_outcome()
        self.assertEqual(disc.neutral_outcome_num, 1)
        disc.neutral_outcome()
        self.assertEqual(disc.neutral_outcome_num, 2)

        # Simulate a successful increase
        self.assertEqual(len(disc.increase), 0)
        self.assertEqual(len(disc.increase_successful), 0)
        self.assertEqual(len(disc.increase_unsuccessful), 0)
        disc.increase_pred(1, True)
        self.assertEqual(len(disc.increase), 1)
        self.assertEqual(len(disc.increase_successful), 1)
        self.assertEqual(len(disc.increase_unsuccessful), 0)
        current_points += disc.reward
        self.assertEqual(disc.points_hist[-1], current_points)

        # Simulate an unsuccessful increase
        disc.increase_pred(0.5, False)
        self.assertEqual(len(disc.increase), 2)
        self.assertEqual(len(disc.increase_successful), 1)
        self.assertEqual(len(disc.increase_unsuccessful), 1)
        current_points -= disc.penalty
        self.assertEqual(disc.points_hist[-1], current_points)

        # Simulate a successful decrease
        self.assertEqual(len(disc.decrease), 0)
        self.assertEqual(len(disc.decrease_successful), 0)
        self.assertEqual(len(disc.decrease_unsuccessful), 0)
        disc.decrease_pred(-1, True)
        self.assertEqual(len(disc.decrease), 1)
        self.assertEqual(len(disc.decrease_successful), 1)
        self.assertEqual(len(disc.decrease_unsuccessful), 0)
        current_points += disc.reward
        self.assertEqual(disc.points_hist[-1], current_points)

        # Simulate an unsuccessful increase
        disc.decrease_pred(-0.5, False)
        self.assertEqual(len(disc.decrease), 2)
        self.assertEqual(len(disc.decrease_successful), 1)
        self.assertEqual(len(disc.decrease_unsuccessful), 1)
        current_points -= disc.penalty
        self.assertEqual(disc.points_hist[-1], current_points)

        # Add a few more fake events
        disc.increase_pred(1, True)
        disc.increase_pred(1, True)
        disc.increase_pred(1, True)
        disc.decrease_pred(-1, True)
        disc.decrease_pred(-1, True)
        disc.decrease_pred(-1, True)
        current_points += disc.reward * 6
        self.assertEqual(disc.points_hist[-1], current_points)

        # Output the discovery and validate its integrity
        d: IDiscovery = disc.build()
        self.assertIsInstance(d, dict)

        # Predictions
        self.assertEqual(d["neutral_num"], 2)
        self.assertEqual(d["increase_num"], 5)
        self.assertEqual(d["decrease_num"], 5)

        # Outcomes
        self.assertEqual(d["neutral_outcome_num"], 2)
        self.assertEqual(d["increase_outcome_num"], 5)
        self.assertEqual(d["decrease_outcome_num"], 5)

        # Points
        self.assertEqual(len(d["points_hist"]), 10)
        self.assertEqual(d["points"], current_points)

        # Accuracy
        self.assertEqual(d["increase_accuracy"], Utils.get_percentage_out_of_total(4, 5))
        self.assertEqual(d["decrease_accuracy"], Utils.get_percentage_out_of_total(4, 5))
        self.assertEqual(d["accuracy"], Utils.get_percentage_out_of_total(8, 10))

        # Increase
        self.assertEqual(len(d["increase_list"]), 5)
        self.assertEqual(d["increase_min"], 0.5)
        self.assertEqual(d["increase_max"], 1)
        self.assertEqual(d["increase_mean"], round(mean([1, 0.5, 1, 1, 1]), 6))
        self.assertEqual(d["increase_median"], round(median([1, 0.5, 1, 1, 1]), 6))

        # Increase Successful
        self.assertEqual(len(d["increase_successful_list"]), 4)
        self.assertEqual(d["increase_successful_mean"], 1)
        self.assertEqual(d["increase_successful_median"], 1)

        # Increase Unsuccessful
        self.assertEqual(len(d["increase_unsuccessful_list"]), 1)
        self.assertEqual(d["increase_unsuccessful_mean"], 0.5)
        self.assertEqual(d["increase_unsuccessful_median"], 0.5)

        # Decrease
        self.assertEqual(len(d["decrease_list"]), 5)
        self.assertEqual(d["decrease_min"], -1)
        self.assertEqual(d["decrease_max"], -0.5)
        self.assertEqual(d["decrease_mean"], round(mean([-1, -0.5, -1, -1, -1]), 6))
        self.assertEqual(d["decrease_median"], round(median([-1, -0.5, -1, -1, -1]), 6))

        # Decrease Successful
        self.assertEqual(len(d["decrease_successful_list"]), 4)
        self.assertEqual(d["decrease_successful_mean"], -1)
        self.assertEqual(d["decrease_successful_median"], -1)

        # Decrease Unsuccessful
        self.assertEqual(len(d["decrease_unsuccessful_list"]), 1)
        self.assertEqual(d["decrease_unsuccessful_mean"], -0.5)
        self.assertEqual(d["decrease_unsuccessful_median"], -0.5)















# Test Execution
if __name__ == '__main__':
    main()
