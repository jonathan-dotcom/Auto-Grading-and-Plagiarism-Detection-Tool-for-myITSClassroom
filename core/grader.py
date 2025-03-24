import os
import tempfile
import subprocess
from typing import Dict, List, Any, Union, Tuple

import docker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseGrader:
    """Base class for all graders."""

    def __init__(self, answer_key: str, total_points: int = 100):
        """Initialize the grader.

        Args:
            answer_key: The answer key to grade against
            total_points: Maximum points possible
        """
        self.answer_key = answer_key
        self.total_points = total_points

    def grade(self, submission: str) -> Dict[str, Any]:
        """Grade a submission.

        Args:
            submission: The submission content

        Returns:
            Dictionary containing grade and feedback
        """
        raise NotImplementedError("Subclasses must implement grade method")


class TextGrader(BaseGrader):
    """Grader for text/essay submissions using NLP similarity."""

    def __init__(self, answer_key: str, total_points: int = 100, threshold: float = 0.7):
        """Initialize the text grader.

        Args:
            answer_key: The answer key to grade against
            total_points: Maximum points possible
            threshold: Similarity threshold for full points
        """
        super().__init__(answer_key, total_points)
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer()

    def grade(self, submission: str) -> Dict[str, Any]:
        """Grade a text submission using semantic similarity.

        Args:
            submission: The submission content

        Returns:
            Dictionary containing grade and feedback
        """
        # Calculate semantic similarity using TF-IDF and cosine similarity
        documents = [self.answer_key, submission]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Calculate grade based on similarity
        grade_percentage = min(1.0, similarity / self.threshold)
        points = round(grade_percentage * self.total_points, 2)

        # Generate feedback
        if similarity >= self.threshold:
            feedback = "Excellent answer, matches expected response well."
        elif similarity >= 0.5:
            feedback = "Good answer, covers many key points but missing some elements."
        else:
            feedback = "Answer needs improvement, missing many key points."

        return {
            'points': points,
            'max_points': self.total_points,
            'percentage': round(grade_percentage * 100, 2),
            'similarity': round(similarity, 4),
            'feedback': feedback
        }


class CodeGrader(BaseGrader):
    """Grader for code submissions using Docker execution and test cases."""

    def __init__(
            self,
            answer_key: str,
            test_cases: List[Dict[str, Any]],
            language: str = 'python',
            total_points: int = 100
    ):
        """Initialize the code grader.

        Args:
            answer_key: Reference implementation
            test_cases: List of test cases to run
            language: Programming language of the submission
            total_points: Maximum points possible
        """
        super().__init__(answer_key, total_points)
        self.test_cases = test_cases
        self.language = language
        self.client = docker.from_env()

    def grade(self, submission: str) -> Dict[str, Any]:
        """Grade a code submission using Docker execution and test cases.

        Args:
            submission: The submission content

        Returns:
            Dictionary containing grade and feedback
        """
        # Create temporary directory for code and test files
        temp_dir = tempfile.mkdtemp()

        try:
            # Set up files based on language
            if self.language == 'python':
                test_results = self._grade_python(submission, temp_dir)
            else:
                # For other languages, implement specific grading logic
                raise NotImplementedError(f"Grading for {self.language} not implemented yet")

            # Calculate overall grade
            passed_tests = sum(1 for result in test_results if result['passed'])
            total_tests = len(test_results)

            if total_tests == 0:
                grade_percentage = 0
            else:
                grade_percentage = passed_tests / total_tests

            points = round(grade_percentage * self.total_points, 2)

            # Generate feedback
            if grade_percentage == 1.0:
                feedback = "Perfect! All test cases passed."
            elif grade_percentage >= 0.7:
                feedback = f"Good job! {passed_tests}/{total_tests} test cases passed."
            else:
                feedback = f"Needs improvement. Only {passed_tests}/{total_tests} test cases passed."

            return {
                'points': points,
                'max_points': self.total_points,
                'percentage': round(grade_percentage * 100, 2),
                'test_results': test_results,
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'feedback': feedback
            }

        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def _grade_python(self, submission: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Grade a Python submission.

        Args:
            submission: The Python code
            temp_dir: Temporary directory for files

        Returns:
            List of test results
        """
        # Create submission file
        submission_path = os.path.join(temp_dir, 'submission.py')
        with open(submission_path, 'w') as f:
            f.write(submission)

        # Create test file
        test_file_content = """
import unittest
import submission

class TestSubmission(unittest.TestCase):
"""
        for i, test_case in enumerate(self.test_cases):
            test_function = f"""
    def test_{i + 1}(self):
        """
            if 'input' in test_case and 'expected_output' in test_case:
                # For function output testing
                test_function += f"""
        result = submission.{test_case.get('function_name', 'main')}({test_case['input']})
        self.assertEqual(result, {test_case['expected_output']})
        """
            elif 'assertion' in test_case:
                # For custom assertions
                test_function += f"""
        {test_case['assertion']}
        """

            test_file_content += test_function

        test_file_content += """
if __name__ == '__main__':
    import json
    import sys
    from unittest import TextTestRunner, TestResult

    class JSONTestResult(TestResult):
        def __init__(self):
            super().__init__()
            self.results = []

        def addSuccess(self, test):
            super().addSuccess(test)
            self.results.append({
                'name': test._testMethodName,
                'passed': True,
                'message': ''
            })

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.results.append({
                'name': test._testMethodName,
                'passed': False,
                'message': str(err[1])
            })

        def addError(self, test, err):
            super().addError(test, err)
            self.results.append({
                'name': test._testMethodName,
                'passed': False,
                'message': str(err[1])
            })

    result = JSONTestResult()
    TextTestRunner(verbosity=2).run(unittest.makeSuite(TestSubmission), result=result)
    print(json.dumps(result.results))
"""

        test_path = os.path.join(temp_dir, 'test_submission.py')
        with open(test_path, 'w') as f:
            f.write(test_file_content)

        # Run tests in Docker container
        try:
            container = self.client.containers.run(
                'python:3.9-slim',
                f'cd /app && python test_submission.py',
                volumes={temp_dir: {'bind': '/app', 'mode': 'ro'}},
                remove=True,
                stdout=True,
                stderr=True
            )
            output = container.decode('utf-8')

            # Parse JSON output
            import json
            import re

            # Find JSON in output
            json_match = re.search(r'\[(.*?)\]', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, return error
                return [{'name': 'execution', 'passed': False, 'message': output}]

        except Exception as e:
            # Handle Docker or execution errors
            return [{'name': 'execution', 'passed': False, 'message': str(e)}]


class GradingManager:
    """Manage the grading process for different types of submissions."""

    def __init__(self):
        """Initialize the grading manager."""
        self.graders = {}

    def register_grader(self, assignment_type: str, grader: BaseGrader):
        """Register a grader for a specific assignment type.

        Args:
            assignment_type: Type of assignment (e.g., 'text', 'code')
            grader: Grader instance
        """
        self.graders[assignment_type] = grader

    def grade_submission(self, assignment_type: str, submission: str) -> Dict[str, Any]:
        """Grade a submission.

        Args:
            assignment_type: Type of assignment
            submission: Submission content

        Returns:
            Grading results
        """
        if assignment_type not in self.graders:
            raise ValueError(f"No grader registered for {assignment_type}")

        return self.graders[assignment_type].grade(submission)