import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.bp_1d_construct.evaluation import BP1DEvaluation
from typing import List, Tuple

# Define the greedy heuristic function
def determine_next_assignment(remaining_items: List[int], remaining_capacities: List[int]) -> Tuple[int, int | None]:
    """
    Determine the next item and bin to pack based on a greedy heuristic.

    Args:
        remaining_items: A list of remaining item weights.
        remaining_capacities: A list of remaining capacities of feasible bins.

    Returns:
        A tuple containing:
        - The selected item to pack.
        - The selected bin to pack the item into (or None if no feasible bin is found).
    """
    # Simple greedy heuristic: choose the largest item that fits into the bin with the smallest remaining capacity
    for item in sorted(remaining_items, reverse=True):  # Try largest items first
        for bin_id, capacity in enumerate(remaining_capacities):
            if item <= capacity:
                return item, bin_id  # Return the selected item and bin
    return remaining_items[0], None  # If no feasible bin is found, return the first item and no bin

def determine_next_assignment(remaining_items: List[int], remaining_capacities: List[int]) -> Tuple[int, int | None]:
    """
    Determine the next item and bin to pack based on a greedy heuristic.

    Args:
        remaining_items: A list of remaining item weights.
        remaining_capacities: A list of remaining capacities of feasible bins.

    Returns:
        A tuple containing:
        - The selected item to pack.
        - The selected bin to pack the item into (or None if no feasible bin is found).
    """
    sorted_items = sorted(remaining_items)
    sorted_bins = sorted(remaining_capacities, reverse=True)

    for item in sorted_items:
        for idx, bin_capacity in enumerate(sorted_bins):
            if bin_capacity >= item:
                return (item, idx)

    return (remaining_items[0], None)

class TestBP1DEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the BP1DEvaluation instance
        bp1d = BP1DEvaluation(
                timeout_seconds = 60,)

        # Call the evaluate_program method
        ave_bins = bp1d.evaluate_program('_', determine_next_assignment)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(ave_bins, float)

        # Optionally, print the result for debugging
        print(f"Average bins used: {ave_bins}")

        # Plot the solution if plot is True
        if plot:
            # Example instance data for plotting
            instance_data = bp1d._datasets[0]
            item_weights, bin_capacity = instance_data

            # Pack items and get the bins
            _, bins = bp1d.pack_items(item_weights, bin_capacity, determine_next_assignment, bp1d.n_bins)

            # Plot the bins
            bp1d.plot_bins(bins, bin_capacity)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestBP1DEvaluation()

    test.test_evaluate_program(plot=plot_solution)
