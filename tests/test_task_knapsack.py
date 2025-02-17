import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.knapsack_construct.evaluation import KnapsackEvaluation
from typing import List, Tuple

def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
    """
    Select the item with the highest value-to-weight ratio that fits in the remaining capacity.

    Args:
        remaining_capacity: The remaining capacity of the knapsack.
        remaining_items: List of tuples containing (weight, value, index) of remaining items.

    Returns:
        The selected item as a tuple (weight, value, index), or None if no item fits.
    """
    best_item = None
    best_ratio = -1  # Initialize with a negative value to ensure any item will have a higher ratio

    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            ratio = value / weight  # Calculate value-to-weight ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item

    return best_item


class TestKnapsackEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the CFLPEvaluation instance
        ks = KnapsackEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = ks.evaluate_program('_', select_next_item)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            instance = ks._datasets[0]
            item_weights, item_values, knapsack_capacity = instance
            _, solution = ks.pack_items(item_weights, item_values, knapsack_capacity, select_next_item)

            # Plot the bins
            ks.plot_solution(item_weights, item_values, solution, knapsack_capacity)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestKnapsackEvaluation()

    test.test_evaluate_program(plot=plot_solution)
