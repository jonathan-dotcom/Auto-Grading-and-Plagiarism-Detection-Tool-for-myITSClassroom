import os
import zipfile
import tempfile
import shutil
import re
from typing import List, Dict, Tuple, Any


class FileProcessor:
    """Process myITS Classroom ZIP files and extract student submissions."""

    def __init__(self, temp_dir: str = None):
        """Initialize the file processor.

        Args:
            temp_dir: Directory to store extracted files temporarily
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()

    def extract_zip(self, zip_path: str) -> str:
        """Extract the ZIP file to a temporary directory.

        Args:
            zip_path: Path to the ZIP file

        Returns:
            Path to the extracted directory
        """
        extract_dir = os.path.join(self.temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        return extract_dir

    def parse_moodle_structure(self, extract_dir: str) -> List[Dict[str, Any]]:
        """Parse the Moodle directory structure to extract student submissions.

        Moodle typically organizes submissions in a structure like:
        - [assignment_name]/
          - [student_full_name]_[id]_assignsubmission_file_/
            - [files...]

        Args:
            extract_dir: Path to the extracted directory

        Returns:
            List of dictionaries containing student info and submission paths
        """
        submissions = []

        # Regular expression to match Moodle's directory naming pattern
        pattern = r'(.+)_(\d+)_assignsubmission_file_'

        for root, dirs, files in os.walk(extract_dir):
            for dir_name in dirs:
                match = re.match(pattern, dir_name)
                if match:
                    student_name = match.group(1)
                    student_id = match.group(2)

                    submission_dir = os.path.join(root, dir_name)
                    submission_files = []

                    # Get all files in the submission directory
                    for sub_root, _, sub_files in os.walk(submission_dir):
                        for file in sub_files:
                            file_path = os.path.join(sub_root, file)
                            submission_files.append({
                                'name': file,
                                'path': file_path,
                                'extension': os.path.splitext(file)[1].lower()
                            })

                    submissions.append({
                        'student_name': student_name,
                        'student_id': student_id,
                        'submission_dir': submission_dir,
                        'files': submission_files
                    })

        return submissions

    def get_file_content(self, file_path: str) -> str:
        """Get the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            Content of the file as a string
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def categorize_submissions(self, submissions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize submissions by file type.

        Args:
            submissions: List of submission dictionaries

        Returns:
            Dictionary with file types as keys and submission lists as values
        """
        categorized = {
            'code': [],
            'text': [],
            'other': []
        }

        code_extensions = ['.py', '.java', '.c', '.cpp', '.js', '.php']
        text_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx', '.odt']

        for submission in submissions:
            for file in submission['files']:
                ext = file['extension']

                if ext in code_extensions:
                    file['type'] = 'code'
                    categorized['code'].append({**submission, 'current_file': file})
                elif ext in text_extensions:
                    file['type'] = 'text'
                    categorized['text'].append({**submission, 'current_file': file})
                else:
                    file['type'] = 'other'
                    categorized['other'].append({**submission, 'current_file': file})

        return categorized

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)