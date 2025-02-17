import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from llm4ad.task.optimization.ovrp_construct.evaluation import OVRPEvaluation
from typing import List, Tuple

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]
    return next_node


class TestovrpEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the ovrpEvaluation instance
        ovrp = OVRPEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = ovrp.evaluate_program('_', select_next_node)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            instance = ovrp._datasets[0]
            instance, distance_matrix, demands, vehicle_capacity = instance
            route = ovrp.route_construct(distance_matrix, demands, vehicle_capacity, select_next_node)

            # Plot the bins
            ovrp.plot_solution(instance, route, demands, vehicle_capacity)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestovrpEvaluation()

    test.test_evaluate_program(plot=plot_solution)
