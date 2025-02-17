import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from llm4ad.task.optimization.admissible_set.evaluation import ASPEvaluation


# Example priority function
def priority(el: tuple, n: int, w: int) -> float:
    """Design a novel algorithm to evaluate a vector for potential inclusion in a set
    Args:
        el: Candidate vectors for the admissible set.
        n: Number of dimensions and the length of a vector.
        w: Weight of each vector.

    Return:
        The priorities of `el`.
    """
    priorities = sum([abs(i) for i in el]) / n
    return priorities

class TestASPEvaluationProgram(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ASPEvaluation(timeout_seconds=60, dimension=15, weight=10)

    def test_evaluate_program(self):
        """Test the evaluate_program method with an example priority function."""
        # Call evaluate_program with the example priority function
        result = self.evaluator.evaluate_program('', priority)

        # Check that the result is an integer (since evaluate() returns an integer)
        self.assertIsInstance(result, int)

        # Check that the result is within the expected bounds
        # (This depends on the specific problem and priority function)
        self.assertGreaterEqual(result, -self.evaluator.Optimal_Set_Length[f"n{self.evaluator.dimension}w{self.evaluator.weight}"])
        self.assertLessEqual(result, 0)

if __name__ == '__main__':
    unittest.main()
