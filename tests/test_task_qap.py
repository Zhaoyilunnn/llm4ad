import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from llm4ad.task.optimization.qap_construct.evaluation import QAPEvaluation
from typing import List, Tuple

def select_next_assignment(current_assignment: List[int], flow_matrix: np.ndarray, distance_matrix: np.ndarray) -> List[int]:
    """
    A greedy heuristic for the Quadratic Assignment Problem.

    Args:
        current_assignment: Current assignment of facilities to locations (-1 means unassigned).
        flow_matrix: Flow matrix between facilities.
        distance_matrix: Distance matrix between locations.

    Returns:
        Updated assignment of facilities to locations.
    """
    n_facilities = len(current_assignment)
    unassigned_facilities = [i for i, loc in enumerate(current_assignment) if loc == -1]
    unassigned_locations = [i for i in range(n_facilities) if i not in current_assignment]

    # If no unassigned facilities or locations, return the current assignment
    if not unassigned_facilities or not unassigned_locations:
        return current_assignment

    # Initialize variables to track the best assignment
    best_facility = None
    best_location = None
    best_cost = float('inf')

    # Evaluate all possible assignments of unassigned facilities to unassigned locations
    for facility in unassigned_facilities:
        for location in unassigned_locations:
            # Calculate the incremental cost of assigning this facility to this location
            incremental_cost = 0
            for assigned_facility, assigned_location in enumerate(current_assignment):
                if assigned_location != -1:
                    incremental_cost += (
                        flow_matrix[facility, assigned_facility] * distance_matrix[location, assigned_location] +
                        flow_matrix[assigned_facility, facility] * distance_matrix[assigned_location, location]
                    )
            # Update the best assignment if this one is better
            if incremental_cost < best_cost:
                best_cost = incremental_cost
                best_facility = facility
                best_location = location

    # Update the assignment
    if best_facility is not None and best_location is not None:
        current_assignment[best_facility] = best_location

    return current_assignment


class TestQAPEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the qapEvaluation instance
        qap = QAPEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = qap.evaluate_program('_', select_next_assignment)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            instance = qap._datasets[0]
            flow_matrix, distance_matrix = instance
            n_facilities = flow_matrix.shape[0]
            current_assignment = [-1] * n_facilities  # Initialize with no assignments
            current_assignment = qap.qap_evaluate(current_assignment, flow_matrix, distance_matrix, select_next_assignment)

            # Plot the bins
            qap.plot_solution(flow_matrix, distance_matrix, current_assignment)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestQAPEvaluation()

    test.test_evaluate_program(plot=plot_solution)
