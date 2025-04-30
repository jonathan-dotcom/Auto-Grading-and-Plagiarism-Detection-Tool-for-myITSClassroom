import os
import tempfile
import subprocess
from typing import Dict, List, Any, Union, Tuple
import re
import json

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


class ShortAnswerGrader(BaseGrader):
    """Grader for short-answer submissions using keyword matching and NLP."""

    def __init__(self, answer_key: str, keywords: List[str], total_points: int = 100,
                 keyword_match_points: int = 70, similarity_points: int = 30):
        """Initialize the short answer grader.

        Args:
            answer_key: The model answer
            keywords: List of essential keywords/concepts that should be present
            total_points: Maximum points possible
            keyword_match_points: Points allocated to keyword matching (out of total)
            similarity_points: Points allocated to overall similarity (out of total)
        """
        super().__init__(answer_key, total_points)
        self.keywords = [k.lower() for k in keywords]
        self.keyword_match_points = keyword_match_points
        self.similarity_points = similarity_points
        self.vectorizer = TfidfVectorizer()

    def grade(self, submission: str) -> Dict[str, Any]:
        """Grade a short answer submission.

        Args:
            submission: The submission content

        Returns:
            Dictionary containing grade and feedback
        """
        # Check for keyword matches
        submission_lower = submission.lower()
        matched_keywords = []
        missing_keywords = []

        for keyword in self.keywords:
            if keyword in submission_lower:
                matched_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        keyword_match_percentage = len(matched_keywords) / len(self.keywords) if self.keywords else 0
        keyword_points = keyword_match_percentage * self.keyword_match_points

        # Calculate semantic similarity
        documents = [self.answer_key, submission]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarity_points = similarity * self.similarity_points

        # Calculate total points
        points = round(keyword_points + similarity_points, 2)

        # Generate feedback
        feedback = f"You included {len(matched_keywords)}/{len(self.keywords)} key concepts. "

        if missing_keywords:
            feedback += f"Missing concepts: {', '.join(missing_keywords)}. "

        if similarity >= 0.8:
            feedback += "Your answer aligns very well with the expected response."
        elif similarity >= 0.6:
            feedback += "Your answer covers the main ideas but could be more precise."
        else:
            feedback += "Your answer needs significant improvement in addressing the question."

        return {
            'points': points,
            'max_points': self.total_points,
            'percentage': round((points / self.total_points) * 100, 2),
            'keyword_match': keyword_match_percentage,
            'similarity': round(similarity, 4),
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords,
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
        """Initialize the code grader."""
        super().__init__(answer_key, total_points)
        self.test_cases = test_cases
        self.language = language

        # Initialize Docker client with Linux-specific connection methods
        self.client = None
        connection_errors = []

        try:
            import platform
            system = platform.system()
            print(f"Detected platform: {system}")

            if system == 'Linux':
                # For Linux servers, try these connection methods
                try:
                    # Clear any problematic environment variables
                    import os
                    if 'DOCKER_HOST' in os.environ:
                        print(f"Found DOCKER_HOST={os.environ['DOCKER_HOST']}, clearing it")
                        saved_docker_host = os.environ['DOCKER_HOST']
                        del os.environ['DOCKER_HOST']
                    else:
                        saved_docker_host = None

                    try:
                        # Try direct socket connection first (most reliable on Linux)
                        import docker
                        print("Trying direct socket connection...")
                        self.client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
                        print("Connected to Docker using Unix socket")
                    except Exception as e:
                        connection_errors.append(f"Unix socket error: {str(e)}")

                        # If that fails, try default connection
                        if not self.client:
                            try:
                                import docker
                                print("Trying default connection...")
                                self.client = docker.from_env()
                                print("Connected to Docker using environment")
                            except Exception as e:
                                connection_errors.append(f"Default connection error: {str(e)}")

                        # If all else fails, try TCP connection
                        if not self.client:
                            try:
                                import docker
                                print("Trying TCP connection...")
                                self.client = docker.DockerClient(base_url='tcp://localhost:2375')
                                print("Connected to Docker using TCP")
                            except Exception as e:
                                connection_errors.append(f"TCP connection error: {str(e)}")

                    # Restore environment if we changed it
                    if saved_docker_host is not None:
                        os.environ['DOCKER_HOST'] = saved_docker_host

                except Exception as e:
                    connection_errors.append(f"Linux connection error: {str(e)}")
            elif system == 'Windows':
                try:
                    import docker
                    self.client = docker.DockerClient(base_url='npipe:////./pipe/docker_engine')
                    print("Connected to Docker using named pipe")
                except Exception as e:
                    connection_errors.append(f"Named pipe error: {str(e)}")

                # If named pipe failed, try TCP connection
                if not self.client:
                    try:
                        import docker
                        self.client = docker.DockerClient(base_url='tcp://localhost:2375')
                        print("Connected to Docker using TCP")
                    except Exception as e:
                        connection_errors.append(f"TCP error: {str(e)}")
            else:  # Mac or other
                try:
                    import docker
                    self.client = docker.from_env()
                    print("Connected to Docker using environment variables")
                except Exception as e:
                    try:
                        import docker
                        self.client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
                        print("Connected to Docker using socket")
                    except Exception as e2:
                        connection_errors.append(f"Unix connection error: {str(e2)}")
        except Exception as e:
            connection_errors.append(f"General error: {str(e)}")

        if not self.client:
            print(f"Failed to connect to Docker. Errors: {connection_errors}")

            # Create a mock Docker client for testing
            class MockDockerContainer:
                def decode(self, encoding):
                    return """
Running tests...
.F.
======================================================================
FAIL: test_2 (__main__.TestSubmission)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_submission.py", line 12, in test_2
    self.assertEqual(result, 10)
AssertionError: 5 != 10

----------------------------------------------------------------------
Ran 3 tests in 0.001s

FAILED (failures=1)
[{"name": "test_1", "passed": true, "message": ""}, {"name": "test_2", "passed": false, "message": "5 != 10"}, {"name": "test_3", "passed": true, "message": ""}]
"""

            class MockDockerContainers:
                def run(self, image, command, volumes, remove, stdout, stderr):
                    print(f"Mock Docker: Running {image} with command {command}")
                    return MockDockerContainer()

            class MockDockerClient:
                def __init__(self):
                    self.containers = MockDockerContainers()

            print("Using mock Docker client for testing")
            self.client = MockDockerClient()

    def grade(self, submission: str) -> Dict[str, Any]:
        """Grade a code submission using Docker execution and test cases."""
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

        # Fixed test runner code
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

    # Create test suite
    suite = unittest.makeSuite(TestSubmission)

    # Create custom result and run tests
    result = JSONTestResult()
    runner = TextTestRunner(verbosity=2)
    runner._makeResult = lambda: result  # Override the _makeResult method
    runner.run(suite)

    # Print the JSON results
    print(json.dumps(result.results))
"""

        test_path = os.path.join(temp_dir, 'test_submission.py')
        with open(test_path, 'w') as f:
            f.write(test_file_content)

        # Check if we have a real Docker client or a mock one
        if not isinstance(self.client, MockDockerClient):
            try:
                # Run tests in Docker container
                container = self.client.containers.run(
                    'python:3.9-slim',
                    ["/bin/sh", "-c", "cd /app && python test_submission.py"],
                    volumes={temp_dir: {'bind': '/app', 'mode': 'ro'}},
                    remove=True,
                    stdout=True,
                    stderr=True
                )
                output = container.decode('utf-8')

                # Parse JSON output
                json_match = re.search(r'\[(.*?)\]', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    # If no JSON found, return error
                    return [{'name': 'execution', 'passed': False, 'message': output}]
            except Exception as e:
                print(f"Docker execution failed: {e}")
                print("Falling back to local Python execution")

                # Use local Python execution as fallback
                return local_python_test_runner(submission, self.test_cases, temp_dir)
        else:
            # If we're using a mock client, use local Python instead for real testing
            print("Using local Python execution instead of mock Docker")
            return local_python_test_runner(submission, self.test_cases, temp_dir)


class GradingManager:
    """Manage the grading process for different types of submissions."""

    def __init__(self):
        """Initialize the grading manager."""
        self.graders = {}

    def register_grader(self, assignment_type: str, grader: BaseGrader):
        """Register a grader for a specific assignment type.

        Args:
            assignment_type: Type of assignment (e.g., 'text', 'code', 'short_answer')
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


# Local Python test runner for fallback when Docker isn't available
def local_python_test_runner(submission: str, test_cases: List[Dict[str, Any]], temp_dir: str = None) -> List[
    Dict[str, Any]]:
    """
    Run Python tests locally without Docker for Linux environments where Docker isn't working.
    This can be used as a fallback when Docker connection fails.

    Args:
        submission: The Python code to test
        test_cases: List of test cases to run
        temp_dir: Optional temporary directory to use (will create one if not provided)

    Returns:
        List of test results in the same format as Docker-based grading
    """
    # Create temporary directory if none provided
    created_temp_dir = False
    if not temp_dir:
        temp_dir = tempfile.mkdtemp()
        created_temp_dir = True

    try:
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
        for i, test_case in enumerate(test_cases):
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

        # Fixed test runner code
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

    # Create test suite
    suite = unittest.makeSuite(TestSubmission)

    # Create custom result and run tests
    result = JSONTestResult()
    runner = TextTestRunner(verbosity=2)
    runner._makeResult = lambda: result  # Override the _makeResult method
    runner.run(suite)

    # Print the JSON results
    print(json.dumps(result.results))
"""

        test_path = os.path.join(temp_dir, 'test_submission.py')
        with open(test_path, 'w') as f:
            f.write(test_file_content)

        # Run tests using local Python interpreter
        print(f"Running tests locally in {temp_dir}")

        # Change to the temp directory
        current_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Run the tests (with timeout for safety)
            result = subprocess.run(['python3', 'test_submission.py'],
                                    capture_output=True,
                                    text=True,
                                    timeout=30)

            output = result.stdout

            # Change back to original directory
            os.chdir(current_dir)

            # Parse JSON output
            json_match = re.search(r'\[(.*?)\]', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, return error
                return [{'name': 'execution', 'passed': False, 'message': output}]

        except subprocess.TimeoutExpired:
            # Change back to original directory
            os.chdir(current_dir)
            return [{'name': 'execution', 'passed': False, 'message': 'Test execution timed out after 30 seconds'}]

        except Exception as e:
            # Change back to original directory
            os.chdir(current_dir)
            return [{'name': 'execution', 'passed': False, 'message': str(e)}]

    finally:
        # Clean up if we created the temp dir
        if created_temp_dir:
            import shutil
            shutil.rmtree(temp_dir)