# grader.py v2 (Fixed Round 2) - Modified for Local C/C++ Compilation
import os
import tempfile
import subprocess
import json
import re
from typing import Dict, List, Any, Union, Tuple
import time
import logging

import docker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logger = logging.getLogger('grading')
docker_logger = logging.getLogger('docker_operations')


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
        self.language = language.lower()

        logger.info(f"Initializing CodeGrader for {self.language}")
        logger.debug(f"Test cases: {len(test_cases)} cases")
        logger.debug(f"Total points: {total_points}")

        # Only initialize Docker for non-C/C++ languages
        if self.language not in ['c', 'cpp']:
            self.client = self._initialize_docker()

    def _initialize_docker(self):
        """Initialize Docker client with fallback options."""
        logger.info("Initializing Docker client")
        connection_errors = []

        try:
            # Try multiple connection methods
            import platform
            system = platform.system()
            logger.debug(f"Operating system: {system}")

            if system == 'Windows':
                logger.debug("Trying Windows Docker connections")
                try:
                    import docker
                    client = docker.DockerClient(base_url='npipe:////./pipe/docker_engine')
                    client.ping()  # Test connection
                    logger.info("Connected to Docker using named pipe")
                    return client
                except Exception as e:
                    connection_errors.append(f"Named pipe error: {str(e)}")
                    logger.debug(f"Named pipe connection failed: {e}")

                try:
                    import docker
                    client = docker.DockerClient(base_url='tcp://localhost:2375')
                    client.ping()
                    logger.info("Connected to Docker using TCP")
                    return client
                except Exception as e:
                    connection_errors.append(f"TCP error: {str(e)}")
                    logger.debug(f"TCP connection failed: {e}")
            else:
                logger.debug("Trying Unix Docker connections")
                try:
                    import docker
                    client = docker.from_env()
                    client.ping()
                    logger.info("Connected to Docker using environment variables")
                    return client
                except Exception as e:
                    connection_errors.append(f"Unix connection error: {str(e)}")
                    logger.debug(f"Environment connection failed: {e}")
        except Exception as e:
            connection_errors.append(f"General error: {str(e)}")
            logger.error(f"General Docker connection error: {e}")

        logger.warning(f"Failed to connect to Docker. Errors: {connection_errors}")
        logger.info("Using mock Docker client for testing")
        return self._create_mock_docker()

    def _create_mock_docker(self):
        """Create mock Docker client for testing."""
        logger.debug("Creating mock Docker client")

        class MockDockerContainer:
            def decode(self, encoding):
                return '[{"name": "test_1", "passed": true, "message": ""}, {"name": "test_2", "passed": false, "message": "Mock test failure"}]'

        class MockDockerContainers:
            def run(self, image, command, volumes, remove, stdout, stderr):
                logger.debug(f"Mock Docker run: {image} with command {command}")
                return MockDockerContainer()

        class MockDockerClient:
            def __init__(self):
                self.containers = MockDockerContainers()

        return MockDockerClient()

    def grade(self, submission: str) -> Dict[str, Any]:
        """Grade a code submission using Docker execution and test cases."""
        start_time = time.time()
        logger.info(f"Starting code grading for {self.language}")
        logger.debug(f"Submission length: {len(submission)} characters")
        logger.debug(f"Test cases: {len(self.test_cases)}")

        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")

        try:
            if self.language == 'python':
                logger.debug("Grading Python submission")
                test_results = self._grade_python(submission, temp_dir)
            elif self.language in ['c', 'cpp']:
                logger.debug(f"Grading {self.language.upper()} submission with local compiler")
                test_results = self._grade_c_cpp_local(submission, temp_dir)
            elif self.language in ['javascript', 'js']:
                logger.debug("Grading JavaScript submission")
                test_results = self._grade_javascript(submission, temp_dir)
            elif self.language == 'java':
                logger.debug("Grading Java submission")
                test_results = self._grade_java(submission, temp_dir)
            else:
                error_msg = f"Grading for {self.language} not implemented yet"
                logger.error(error_msg)
                raise NotImplementedError(error_msg)

            logger.debug(f"Test results: {test_results}")

            # Calculate overall grade
            passed_tests = sum(1 for result in test_results if result['passed'])
            total_tests = len(test_results)

            if total_tests == 0:
                grade_percentage = 0
                logger.warning("No test cases were executed")
            else:
                grade_percentage = passed_tests / total_tests
                logger.info(f"Tests passed: {passed_tests}/{total_tests} ({grade_percentage:.2%})")

            points = round(grade_percentage * self.total_points, 2)

            # Generate feedback
            if grade_percentage == 1.0:
                feedback = "Perfect! All test cases passed."
            elif grade_percentage >= 0.7:
                feedback = f"Good job! {passed_tests}/{total_tests} test cases passed."
            else:
                feedback = f"Needs improvement. Only {passed_tests}/{total_tests} test cases passed."

            execution_time = time.time() - start_time
            logger.info(
                f"Grading completed in {execution_time:.2f} seconds - Final score: {points}/{self.total_points}")

            return {
                'points': points,
                'max_points': self.total_points,
                'percentage': round(grade_percentage * 100, 2),
                'test_results': test_results,
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'feedback': feedback,
                'execution_time': execution_time
            }

        except Exception as e:
            logger.error(f"Error during grading: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Clean up temporary directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

    def _grade_python(self, submission: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Grade a Python submission."""
        submission_path = os.path.join(temp_dir, 'submission.py')
        with open(submission_path, 'w') as f:
            f.write(submission)

        test_file_content = """
import unittest
import json
import sys
import submission

class TestSubmission(unittest.TestCase):
"""
        for i, test_case in enumerate(self.test_cases):
            test_function = f"""
    def test_{i + 1}(self):
        """
            if 'input' in test_case and 'expected_output' in test_case:
                test_function += f"""
        result = submission.{test_case.get('function_name', 'main')}({test_case['input']})
        self.assertEqual(result, {test_case['expected_output']})
        """
            elif 'assertion' in test_case:
                test_function += f"""
        {test_case['assertion']}
        """

            test_file_content += test_function

        test_file_content += """
if __name__ == '__main__':
    import json
    from unittest import TextTestRunner, TestResult

    class JSONTestResult(TestResult):
        def __init__(self):
            super().__init__()
            self.results = []

        def addSuccess(self, test):
            super().addSuccess(test)
            self.results.append({'name': test._testMethodName, 'passed': True, 'message': ''})

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.results.append({'name': test._testMethodName, 'passed': False, 'message': str(err[1])})

        def addError(self, test, err):
            super().addError(test, err)
            self.results.append({'name': test._testMethodName, 'passed': False, 'message': str(err[1])})

    suite = unittest.makeSuite(TestSubmission)
    result = JSONTestResult()
    runner = TextTestRunner(verbosity=0)
    runner._makeResult = lambda: result
    runner.run(suite)
    print(json.dumps(result.results))
"""

        test_path = os.path.join(temp_dir, 'test_submission.py')
        with open(test_path, 'w') as f:
            f.write(test_file_content)

        return self._run_docker_test('python:3.9-slim',
                                     ["/bin/sh", "-c", "cd /app && python test_submission.py"],
                                     temp_dir)

    def _grade_c_cpp_local(self, submission: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Grade a C/C++ submission using local compiler."""
        is_cpp = self.language == 'cpp'
        file_ext = '.cpp' if is_cpp else '.c'
        compiler = 'g++' if is_cpp else 'gcc'

        logger.info(f"Grading {self.language.upper()} submission locally")
        logger.debug(f"Compiler: {compiler}, Extension: {file_ext}")

        # Check if compiler is available
        try:
            subprocess.run([compiler, '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"{compiler} not found on system")
            return [{'name': 'compiler_check', 'passed': False, 'message': f'{compiler} not found on system'}]

        # Write the original submission file
        submission_path = os.path.join(temp_dir, f'submission{file_ext}')
        with open(submission_path, 'w') as f:
            f.write(submission)

        logger.debug(f"Written original submission to: {submission_path}")

        # Analyze submission to extract function information
        functions_info = self._analyze_c_cpp_submission(submission, is_cpp)
        logger.debug(f"Detected functions: {functions_info}")

        # Create modified submission that renames main
        modified_submission_path = os.path.join(temp_dir, f'submission_modified{file_ext}')
        modified_submission = re.sub(r'int\s+main\s*\(\s*\)', 'int student_main()', submission)
        with open(modified_submission_path, 'w') as f:
            f.write(modified_submission)

        # Create test file
        test_content = self._create_c_cpp_test_file(functions_info, is_cpp)
        test_path = os.path.join(temp_dir, f'test{file_ext}')
        with open(test_path, 'w') as f:
            f.write(test_content)

        logger.debug(f"Created test file: {test_path}")

        # Compile the test program
        compiler_flags = ['-std=c++11' if is_cpp else '-std=c11', '-lm']
        executable_path = os.path.join(temp_dir, 'test_program')

        try:
            compile_result = subprocess.run(
                [compiler] + compiler_flags + ['-o', executable_path, test_path],
                cwd=temp_dir,
                capture_output=True,
                text=True
            )

            if compile_result.returncode != 0:
                logger.error(f"Compilation failed: {compile_result.stderr}")
                return [{
                    'name': 'compilation',
                    'passed': False,
                    'message': f"Compilation error: {compile_result.stderr[:500]}"
                }]

            logger.info("Compilation successful")

            # Run the test program
            run_result = subprocess.run(
                [executable_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )

            output = run_result.stdout
            logger.debug(f"Program output: {output}")

            # Parse JSON output
            json_matches = re.findall(r'\[.*?\]', output, re.DOTALL)

            if json_matches:
                json_str = json_matches[-1].strip()
                logger.debug(f"Found JSON output: {json_str}")

                try:
                    result = json.loads(json_str)
                    logger.info(f"Successfully parsed {len(result)} test results")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    return [{
                        'name': 'json_parse_error',
                        'passed': False,
                        'message': f"JSON parse error: {str(e)}"
                    }]
            else:
                logger.warning("No JSON output found")
                return [{
                    'name': 'execution',
                    'passed': False,
                    'message': f"No JSON output found. Raw output: {output[:500]}"
                }]

        except subprocess.TimeoutExpired:
            logger.error("Program execution timed out")
            return [{
                'name': 'timeout',
                'passed': False,
                'message': "Program execution timed out (10 seconds)"
            }]
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            return [{
                'name': 'execution',
                'passed': False,
                'message': f"Execution error: {str(e)}"
            }]

    def _analyze_c_cpp_submission(self, submission: str, is_cpp: bool) -> Dict[str, Any]:
        """Analyze C/C++ submission to extract function information without modifying it."""
        import re

        functions = []
        classes = []

        if is_cpp:
            # Extract C++ class methods
            class_pattern = r'class\s+(\w+).*?\{(.*?)\};'
            class_matches = re.findall(class_pattern, submission, re.DOTALL)

            for class_name, class_body in class_matches:
                # Look for static methods in the class
                method_pattern = r'static\s+(\w+)\s+(\w+)\s*\([^)]*\)'
                method_matches = re.findall(method_pattern, class_body)

                for return_type, method_name in method_matches:
                    functions.append({
                        'name': method_name,
                        'return_type': return_type,
                        'class': class_name,
                        'call_format': f'{class_name}::{method_name}'
                    })

                classes.append(class_name)

        # Extract regular C functions (works for both C and C++)
        # Improved pattern to better capture return types including pointers and const
        function_pattern = r'(?:^|\n)\s*(?:static\s+)?(?:inline\s+)?(?:const\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*\{'
        function_matches = re.findall(function_pattern, submission, re.MULTILINE)

        for return_type, func_name in function_matches:
            if func_name != 'main':  # Skip main function
                # Clean up return type (remove extra spaces)
                return_type = return_type.strip()
                functions.append({
                    'name': func_name,
                    'return_type': return_type,
                    'class': None,
                    'call_format': func_name
                })

        return {
            'functions': functions,
            'classes': classes,
            'is_cpp': is_cpp,
            'has_main': 'main' in submission
        }

    def _create_c_cpp_test_file(self, functions_info: Dict[str, Any], is_cpp: bool) -> str:
        """Create test file that works with original submission by including it."""

        submission_filename = "submission_modified.cpp" if is_cpp else "submission_modified.c"

        # Basic includes
        includes = """#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
"""

        if is_cpp:
            includes += """#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;
"""
        # Directly include the student's code file
        includes += f'#include "{submission_filename}"\n\n'

        # Test runner main function
        test_main = """
// Test results tracking
int passed_tests = 0;
int total_tests = 0;

// Helper to escape JSON strings
void print_escaped(const char* s) {
    while (*s) {
        if (*s == '"' || *s == '\\\\') {
            putchar('\\\\');
        }
        putchar(*s);
        s++;
    }
}

// Helper for floating point comparison with tolerance
int double_equals(double a, double b, double tolerance) {
    double diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < tolerance;
}

void print_test_result(const char* test_name, int passed, const char* message) {
    if (total_tests > 0) printf(", ");
    printf("{\\"name\\": \\"%s\\", \\"passed\\": %s, \\"message\\": \\"", test_name, passed ? "true" : "false");
    print_escaped(message);
    printf("\\"}");
    if (passed) passed_tests++;
    total_tests++;
}

int main() {
    printf("[");

"""

        # Add test cases
        for i, test_case in enumerate(self.test_cases):
            test_name = f"test_{i + 1}"

            if 'function_call' in test_case and 'expected_output' in test_case:
                adapted_call = self._adapt_function_call(test_case['function_call'], functions_info)
                expected_output = test_case['expected_output']

                # C++ bools are true/false, C bools are 1/0. Let's standardize.
                if str(expected_output).lower() == 'true': expected_output = '1'
                if str(expected_output).lower() == 'false': expected_output = '0'

                test_main += f"""
    // Test {i + 1}: {test_case.get('description', 'Test case')}
    {{
        char message[256] = "";
        """

                if is_cpp:
                    test_main += f"""
        try {{
            auto result = {adapted_call};
            auto expected = {expected_output};

            if (result == expected) {{
                print_test_result("{test_name}", 1, "");
            }} else {{
                std::stringstream ss;
                ss << "Expected " << expected << ", got " << result;
                snprintf(message, sizeof(message), "%s", ss.str().c_str());
                print_test_result("{test_name}", 0, message);
            }}
        }} catch (...) {{
            print_test_result("{test_name}", 0, "Function call threw an exception");
        }}
"""
                else:
                    # For C, we need to determine the return type from the test case or function info
                    # Check if test case specifies return type, otherwise infer it
                    if 'return_type' in test_case:
                        return_type = test_case['return_type']
                    else:
                        return_type = self._get_return_type_for_function(adapted_call, functions_info)

                        # If still not found, infer from expected output
                        if return_type == "int" and isinstance(expected_output, (float, str)) and '.' in str(
                                expected_output):
                            return_type = "double"

                    # Check if we're dealing with floating point
                    is_float_type = return_type in ["float", "double"]

                    test_main += f"""
        {return_type} result = {adapted_call};
        {return_type} expected = {expected_output};

        int test_passed = 0;
        """

                    if is_float_type:
                        test_main += f"""
        // Use tolerance-based comparison for floating point
        test_passed = double_equals((double)result, (double)expected, 0.0001);

        if (test_passed) {{
            print_test_result("{test_name}", 1, "");
        }} else {{
            snprintf(message, sizeof(message), "Expected %.6f, got %.6f", (double)expected, (double)result);
            print_test_result("{test_name}", 0, message);
        }}
"""
                    else:
                        test_main += f"""
        test_passed = (result == expected);

        if (test_passed) {{
            print_test_result("{test_name}", 1, "");
        }} else {{
            if (strcmp("{return_type}", "char") == 0) {{
                snprintf(message, sizeof(message), "Expected '%c', got '%c'", expected, result);
            }} else if (strcmp("{return_type}", "long") == 0 || strcmp("{return_type}", "long long") == 0) {{
                snprintf(message, sizeof(message), "Expected %lld, got %lld", (long long)expected, (long long)result);
            }} else {{
                snprintf(message, sizeof(message), "Expected %d, got %d", (int)expected, (int)result);
            }}
            print_test_result("{test_name}", 0, message);
        }}
"""
                test_main += "    }\n"
            else:
                test_main += f"""
    print_test_result("{test_name}", 0, "Invalid test case format");
"""

        test_main += """
    printf("]\\n");
    return 0;
}
"""
        return includes + test_main

    def _adapt_function_call(self, function_call: str, functions_info: Dict[str, Any]) -> str:
        """Adapt function calls to work with the actual submission."""

        # Extract function name from the call
        import re
        func_match = re.match(r'(\w+)\s*\(', function_call)
        if not func_match:
            return function_call

        func_name = func_match.group(1)

        # Find the actual function in the submission
        for func in functions_info['functions']:
            if func['name'] == func_name or func['name'].lower() == func_name.lower():
                if func['class']:
                    # It's a class method, use the class::method format
                    return function_call.replace(func_name, func['call_format'])
                else:
                    # It's a regular function, use as-is
                    return function_call

        # If function not found, try common variations
        variations = {
            'is_prime': ['isPrime', 'is_prime', 'prime'],
            'isPrime': ['is_prime', 'isPrime', 'prime']
        }

        if func_name in variations:
            for func in functions_info['functions']:
                if func['name'] in variations[func_name]:
                    if func['class']:
                        return function_call.replace(func_name, func['call_format'])
                    else:
                        return function_call.replace(func_name, func['name'])

        return function_call

    def _get_return_type_for_function(self, function_call: str, functions_info: Dict[str, Any]) -> str:
        """Get the return type of a function from the function call and info."""
        import re

        # Extract function name from the call
        func_match = re.match(r'(\w+)\s*\(', function_call)
        if not func_match:
            return "int"  # Default to int

        func_name = func_match.group(1)

        # Look for the function in our analyzed functions
        for func in functions_info['functions']:
            if func['name'] == func_name or func['call_format'] == func_name:
                return func.get('return_type', 'int')

        # Common function return types based on naming patterns
        if 'is_' in func_name.lower() or 'check' in func_name.lower():
            return "int"  # boolean functions typically return int in C
        elif 'sqrt' in func_name.lower() or 'average' in func_name.lower():
            return "double"
        else:
            return "int"  # Default to int

    def _grade_javascript(self, submission: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Grade a JavaScript submission."""
        submission_path = os.path.join(temp_dir, 'submission.js')
        with open(submission_path, 'w') as f:
            f.write(submission)

        test_file_content = """
const submission = require('./submission.js');

const results = [];
let testCount = 0;

function assertEqual(expected, actual, testName) {
    testCount++;
    // FIX: Use type-insensitive comparison by converting both to strings.
    // This handles cases like `8` (number) vs `"8"` (string).
    if (String(expected) == String(actual)) {
        results.push({name: testName, passed: true, message: ''});
    } else {
        // Enclose results in quotes for clarity in the message
        const expectedStr = typeof expected === 'string' ? `"${expected}"` : expected;
        const actualStr = typeof actual === 'string' ? `"${actual}"` : actual;
        results.push({name: testName, passed: false, message: `Expected ${expectedStr}, got ${actualStr}`});
    }
}

// Test cases
"""

        for i, test_case in enumerate(self.test_cases):
            if 'function_call' in test_case and 'expected_output' in test_case:
                js_call = f"submission.{test_case['function_call']}"
                expected_output_str = json.dumps(test_case['expected_output'])

                test_file_content += f"""
try {{
    assertEqual({expected_output_str}, {js_call}, 'test_{i + 1}');
}} catch (e) {{
    results.push({{name: 'test_{i + 1}', passed: false, message: e.message}});
}}
"""

        test_file_content += """
console.log(JSON.stringify(results));
"""

        test_path = os.path.join(temp_dir, 'test.js')
        with open(test_path, 'w') as f:
            f.write(test_file_content)

        return self._run_docker_test('node:16-alpine',
                                     ["/bin/sh", "-c", "cd /app && node test.js"],
                                     temp_dir)

    def _grade_java(self, submission: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Grade a Java submission."""
        # Write submission file
        submission_path = os.path.join(temp_dir, 'Submission.java')
        with open(submission_path, 'w') as f:
            f.write(submission)

        # Create test runner that uses the submission
        test_file_content = f"""
import java.util.*;

public class TestRunner {{
    public static void main(String[] args) {{
        System.out.print("[");
        boolean first = true;
"""

        for i, test_case in enumerate(self.test_cases):
            if 'method_call' in test_case and 'expected_output' in test_case:
                test_file_content += f"""
        if (!first) System.out.print(", ");
        first = false;
        try {{
            Object result = Submission.{test_case['method_call']};
            Object expected = {test_case['expected_output']};
            boolean passed = result.equals(expected);
            String message = passed ? "" : "Expected " + expected + ", got " + result;
            message = message.replace("\\"", "\\\\\\""); // Escape quotes
            System.out.printf("{{\\"name\\": \\"test_{i + 1}\\", \\"passed\\": %s, \\"message\\": \\"%s\\"}}",
                passed ? "true" : "false", message);
        }} catch (Exception e) {{
            String errorMessage = e.getMessage().replace("\\"", "\\\\\\"");
            System.out.printf("{{\\"name\\": \\"test_{i + 1}\\", \\"passed\\": false, \\"message\\": \\"Exception: %s\\"}}",
                errorMessage);
        }}
"""

        test_file_content += """
        System.out.println("]");
    }
}
"""

        test_path = os.path.join(temp_dir, 'TestRunner.java')
        with open(test_path, 'w') as f:
            f.write(test_file_content)

        return self._run_docker_test('openjdk:11-jdk-slim',
                                     ["/bin/sh", "-c", "cd /app && javac *.java 2>&1 && java TestRunner 2>&1"],
                                     temp_dir)

    def _run_docker_test(self, image: str, command: List[str], temp_dir: str) -> List[Dict[str, Any]]:
        """Run test in Docker container and parse results."""
        logger.info(f"Running Docker test with image: {image}")
        logger.debug(f"Command: {' '.join(command)}")
        logger.debug(f"Volume mount: {temp_dir}:/app")

        try:
            start_time = time.time()

            # Log files in temp_dir for debugging
            logger.debug("Files in temporary directory before execution:")
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    logger.debug(f"  {file} ({size} bytes)")

            container = self.client.containers.run(
                image,
                command,
                volumes={temp_dir: {'bind': '/app', 'mode': 'rw'}},  # Use rw for java compilation
                remove=True,
                stdout=True,
                stderr=True
            )

            execution_time = time.time() - start_time
            output = container.decode('utf-8')

            logger.info(f"Docker execution completed in {execution_time:.2f} seconds")
            logger.debug(f"Docker output (first 500 chars): {output[:500]}")

            # Log to Docker operations logger
            docker_logger.info(f"DOCKER RUN: {image}")
            docker_logger.debug(f"Command: {' '.join(command)}")
            docker_logger.debug(f"Output: {output}")

            # Parse JSON output
            json_match = re.search(r'\[(.*?)\]', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Found JSON output: {json_str}")
                try:
                    result = json.loads(json_str)
                    logger.info(f"Successfully parsed {len(result)} test results")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    logger.error(f"JSON string was: {json_str}")
                    return [{'name': 'json_parse_error', 'passed': False, 'message': f"JSON parse error: {str(e)}"}]
            else:
                logger.warning("No JSON output found in Docker response")
                logger.debug(f"Full output was: {output}")
                return [
                    {'name': 'execution', 'passed': False, 'message': f"No JSON output found. Raw output: {output}"}]

        except Exception as e:
            logger.error(f"Docker execution error: {str(e)}")
            docker_logger.error(f"DOCKER ERROR: {str(e)}")
            import traceback
            logger.error(f"Docker error traceback: {traceback.format_exc()}")
            return [{'name': 'execution', 'passed': False, 'message': f"Docker execution error: {str(e)}"}]


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