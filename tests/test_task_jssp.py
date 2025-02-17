import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.jssp_construct.evaluation import JSSPEvaluation
from typing import List, Tuple

def determine_next_operation(current_status, feasible_operations):
    """
    Determine the next operation to schedule based on a greedy heuristic.

    Args:
        current_status: A dictionary representing the current status of each machine and job.
        feasible_operations: A list of feasible operations that can be scheduled next.

    Returns:
        The next operation to schedule, represented as a tuple (job_id, machine_id, processing_time).
    """
    # Simple greedy heuristic: choose the operation with the shortest processing time
    next_operation = min(feasible_operations, key=lambda x: x[2])
    return next_operation


class TestJSSPEvaluation(unittest.TestCase):
    
    def test_evaluate_program(self, plot: bool = False):
        """
        Test the evaluate_program method with a simple greedy heuristic.

        Args:
            plot: If True, plot the bin packing solution.
        """

        # Initialize the CFLPEvaluation instance
        jssp = JSSPEvaluation(
                timeout_seconds = 60)

        # Call the evaluate_program method
        average_cost = jssp.evaluate_program('_', determine_next_operation)

        # Assert that the result is a float (since evaluate_program returns a float)
        self.assertIsInstance(average_cost, float)

        # Optionally, print the result for debugging
        print(f"Average cost: {average_cost}")

        # Plot the solution if plot is True
        if plot:
            instance = jssp._datasets[0]
            processing_times, n1, n2 = instance
            makespan,solution  = jssp.schedule_jobs(processing_times, n1, n2,determine_next_operation)

            # Plot the bins
            jssp.plot_solution(solution, n1, n2)

if __name__ == '__main__':
    # Run the test with plotting enabled
    plot_solution = True  # Set to False to disable plotting
    
    test = TestJSSPEvaluation()

    test.test_evaluate_program(plot=plot_solution)
