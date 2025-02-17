import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from llm4ad.task.optimization.set_cover_construct.evaluation import SCPEvaluation
from typing import List, Tuple

def select_next_subset(selected_subsets: List[List[int]], remaining_subsets: List[List[int]], remaining_elements: List[int]) -> List[int] | None:
    """
    A heuristic for the Set Covering Problem.

    Args:
        selected_subsets: List of already selected subsets.
        remaining_subsets: List of remaining subsets to choose from.
        remaining_elements: List of elements still to be covered.

    Returns:
        The next subset to select, or None if no subset can cover any remaining elements.
    """
    max_covered = 0
    best_subset = None

    for subset in remaining_subsets:
        # Calculate the number of uncovered elements this subset covers
        covered = len(set(subset).intersection(remaining_elements))
        if covered > max_covered:
            max_covered = covered
            best_subset = subset

    return best_subset


class TestSCPEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the SCPEvaluation instance
        SCP = SCPEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = SCP.evaluate_program('_', select_next_subset)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            instance = SCP._datasets[0]
            universal_set, subsets = instance
            num_subsets, solution = SCP.cover_subsets(universal_set, subsets, select_next_subset)

            # Plot the bins
            SCP.plot_solution(universal_set, solution, subsets)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestSCPEvaluation()

    test.test_evaluate_program(plot=plot_solution)
