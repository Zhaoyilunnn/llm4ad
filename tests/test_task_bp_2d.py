import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.bp_2d_construct.evaluation import BP2DEvaluation
from typing import List, Tuple

# def determine_next_assignment(remaining_items: List[Tuple[int, int]], feasible_corners: List[List[Tuple[int, int]]]) -> Tuple[Tuple[int, int], int]:
#     """
#     A simple heuristic function to select the next item and bin for 2D bin packing.

#     Args:
#         remaining_items: A list of tuples representing the (width, height) of remaining items.
#         feasible_corners: A list of lists, where each inner list contains the feasible corners for a bin.

#     Returns:
#         A tuple containing:
#         - The selected item (width, height).
#         - The index of the selected bin (or None if no bin is feasible).
#     """
#     # Step 1: Select the largest item by area
#     selected_item = max(remaining_items, key=lambda x: x[0] * x[1])

#     # Step 2: Select the bin with the most feasible corners
#     max_corners = -1
#     selected_bin = None
#     for i, corners in enumerate(feasible_corners):
#         if len(corners) > max_corners:
#             max_corners = len(corners)
#             selected_bin = i

#     # If no bin has feasible corners, return None for the bin
#     if max_corners == 0:
#         selected_bin = None

#     return selected_item, selected_bin
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
        for bin_idx, capacity in enumerate(sorted_bins):
            if item <= capacity:
                return item, bin_idx
    return (sorted_items[0], None) if sorted_items else (None, None)




class TestBP2DEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the BP2DEvaluation instance
        bp2d = BP2DEvaluation(
                timeout_seconds = 60,
                # n_bins = 10,
                # n_instance  = 5,
                # n_items = 10,
                # bin_width = 100,
                # bin_height = 100
                )

        # Call the evaluate_program method
        ave_bins = bp2d.evaluate_program('_', determine_next_assignment)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(ave_bins, float)

        # Optionally, print the result for debugging
        print(f"Average bins used: {ave_bins}")

        # Plot the solution if plot is True
        if plot:
            # Example instance data for plotting
            instance_data = bp2d._datasets[0]
            item_dimensions, bin_dimensions = instance_data

            # Pack items and get the bins
            _, bins = bp2d.pack_items_2d(item_dimensions, bin_dimensions, determine_next_assignment, bp2d.n_bins)

            # Plot the bins
            bp2d.plot_solution(bins, bin_dimensions)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestBP2DEvaluation()

    test.test_evaluate_program(plot=plot_solution)
