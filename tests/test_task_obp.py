import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.online_bin_packing.evaluation import OBPEvaluation
import numpy as np

def priority(item: float, valid_bins: np.ndarray) -> np.ndarray:
    """
    Priority function for the First-Fit Decreasing (FFD) heuristic.

    Args:
        item: The size of the item to be packed.
        valid_bins: A numpy array of remaining capacities in valid bins.

    Returns:
        A numpy array of priorities for the valid bins.
    """
    # Prioritize bins with the least remaining capacity (but still able to fit the item)
    priorities = -valid_bins  # Negative because we want to maximize the priority for the smallest remaining capacity
    return priorities


class TestOBPEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the CFLPEvaluation instance
        obp = OBPEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = obp.evaluate_program('_', priority)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            name = list(obp._datasets.keys())[0]
            instance = obp._datasets[name]
            capacity = instance['capacity']
            items = instance['items']
            # Create num_items bins so there will always be space for all items,
            # regardless of packing order. Array has shape (num_items,).
            bins = np.array([capacity for _ in range(instance['num_items'])])
            # Pack items into bins and return remaining capacity in bins_packed, which
            # has shape (num_items,).
            _, bins_packed = obp.online_binpack(items, bins, priority)

            # Plot the bins
            obp.plot_solution(bins_packed, items, capacity, max_unused_bins=5)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestOBPEvaluation()

    test.test_evaluate_program(plot=plot_solution)
