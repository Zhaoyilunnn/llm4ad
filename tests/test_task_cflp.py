import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.cflp_construct.evaluation import CFLPEvaluation
from typing import List, Tuple

def determine_next_assignment(assignments: List[List[int]], remaining_customers: List[int], remaining_capacities: List[int], customer_demands: List[int], assignment_costs: List[List[int]]) -> Tuple[int, int]:
    """
    Constructive heuristic for the Capacitated Facility Location Problem.
    Assigns the next customer to the facility with the lowest cost that has sufficient capacity.

    Args:
        assignments: Current assignments of customers to facilities.
        remaining_customers: List of customer indices not yet assigned.
        remaining_capacities: Remaining capacities of facilities.
        customer_demands: List of customer demands.
        assignment_costs: 2D list of assignment costs (facility-to-customer).

    Returns:
        A tuple containing:
        - The selected customer index.
        - The selected facility index (or None if no feasible assignment exists).
    """
    # Iterate over all remaining customers
    for customer in remaining_customers:
        # Iterate over all facilities to find the one with the lowest cost and sufficient capacity
        min_cost = float('inf')
        selected_facility = None

        for facility in range(len(remaining_capacities)):
            if remaining_capacities[facility] >= customer_demands[customer] and assignment_costs[facility][customer] < min_cost:
                min_cost = assignment_costs[facility][customer]
                selected_facility = facility

        # If a feasible facility is found, return the customer and facility
        if selected_facility is not None:
            return customer, selected_facility

    # If no feasible assignment is found, return None
    return None, None


class TestCFLPEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the CFLPEvaluation instance
        cflp = CFLPEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = cflp.evaluate_program('_', determine_next_assignment)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            instance = cflp._datasets[0]
            facility_capacities = instance["facility_capacities"]
            customer_demands = instance["customer_demands"]
            assignment_costs = instance["assignment_costs"]
            _, solution = cflp.assign_customers(facility_capacities, customer_demands, assignment_costs, determine_next_assignment)

            # Plot the bins
            cflp.plot_solution(facility_capacities, customer_demands, solution, assignment_costs)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestCFLPEvaluation()

    test.test_evaluate_program(plot=plot_solution)
