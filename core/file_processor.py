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

        Moodle typically organizes submissions in structures like:
        - File submissions: [student_full_name]_[id]_assignsubmission_file_/
        - Online text: [student_full_name]_[id]_assignsubmission_onlinetext_/

        Args:
            extract_dir: Path to the extracted directory

        Returns:
            List of dictionaries containing student info and submission paths
        """
        submissions = []

        # Regular expressions to match Moodle's directory naming patterns
        file_pattern = r'(.+)_(\d+)_assignsubmission_file_'
        onlinetext_pattern = r'(.+)_(\d+)_assignsubmission_onlinetext_'

        for root, dirs, files in os.walk(extract_dir):
            # Process file submissions
            for dir_name in dirs:
                file_match = re.match(file_pattern, dir_name)
                if file_match:
                    student_name = file_match.group(1)
                    student_id = file_match.group(2)

                    submission_dir = os.path.join(root, dir_name)
                    submission_files = []

                    # Get all files in the submission directory
                    for sub_root, _, sub_files in os.walk(submission_dir):
                        for file in sub_files:
                            file_path = os.path.join(sub_root, file)
                            file_ext = os.path.splitext(file)[1].lower()
                            submission_files.append({
                                'name': file,
                                'path': file_path,
                                'extension': file_ext,
                                'type': self._determine_file_type(file_ext)
                            })

                    if submission_files:
                        submissions.append({
                            'student_name': student_name,
                            'student_id': student_id,
                            'submission_dir': submission_dir,
                            'files': submission_files,
                            'submission_type': 'file'
                        })

                # Process online text submissions
                onlinetext_match = re.match(onlinetext_pattern, dir_name)
                if onlinetext_match:
                    student_name = onlinetext_match.group(1)
                    student_id = onlinetext_match.group(2)

                    submission_dir = os.path.join(root, dir_name)
                    online_text_files = []

                    # Look for onlinetext.html files
                    for sub_root, _, sub_files in os.walk(submission_dir):
                        for file in sub_files:
                            if file == 'onlinetext.html':
                                file_path = os.path.join(sub_root, file)
                                text_content = self.extract_text_from_html(file_path)

                                online_text_files.append({
                                    'name': file,
                                    'path': file_path,
                                    'extension': '.html',
                                    'content': text_content,
                                    'type': 'text'
                                })

                    if online_text_files:
                        submissions.append({
                            'student_name': student_name,
                            'student_id': student_id,
                            'submission_dir': submission_dir,
                            'files': online_text_files,
                            'submission_type': 'onlinetext'
                        })

        return submissions

    def _determine_file_type(self, extension: str) -> str:
        """Determine file type based on extension.

        Args:
            extension: File extension

        Returns:
            File type category (code, text, or other)
        """
        code_extensions = ['.py', '.java', '.c', '.cpp', '.js', '.php', '.html', '.css']
        text_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx', '.odt', '.rtf']

        if extension in code_extensions:
            return 'code'
        elif extension in text_extensions:
            return 'text'
        else:
            return 'other'

    def get_file_content(self, file_path: str) -> str:
        """Get the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            Content of the file as a string
        """
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()

        # Extract content based on file type
        if ext in ['.txt', '.md', '.py', '.java', '.c', '.cpp', '.js', '.php', '.html', '.css']:
            # Simple text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            # Try using the extract_text_from_file method for other formats
            return self.extract_text_from_file(file_path)

    def extract_text_from_html(self, file_path: str) -> str:
        """Extract text content from HTML file.

        Args:
            file_path: Path to the HTML file

        Returns:
            Extracted text content
        """
        try:
            # Try using BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text(separator=' ')
            except ImportError:
                # Fallback if BeautifulSoup is not available
                import re
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()

                # Simple regex to remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', html_content)
                return text
        except Exception as e:
            return f"Error extracting HTML content: {str(e)}"

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif ext == '.docx':
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                return "Error: python-docx package required for DOCX processing"

        elif ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join([page.extract_text() for page in reader.pages])
            except ImportError:
                return "Error: PyPDF2 package required for PDF processing"

        elif ext == '.odt':
            try:
                import odf.opendocument
                import odf.text
                doc = odf.opendocument.load(file_path)
                allparas = doc.getElementsByType(odf.text.P)
                return "\n".join([para.firstChild for para in allparas if para.firstChild])
            except ImportError:
                return "Error: odfpy package required for ODT processing"

        elif ext == '.rtf':
            try:
                import striprtf.striprtf
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    rtf_text = f.read()
                return striprtf.striprtf.rtf_to_text(rtf_text)
            except ImportError:
                return "Error: striprtf package required for RTF processing"

        else:
            return f"Unsupported file format: {ext}"

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

        for submission in submissions:
            # Handle online text submissions
            if submission['submission_type'] == 'onlinetext':
                # Online text is always categorized as text
                for file in submission['files']:
                    categorized['text'].append({**submission, 'current_file': file})
                continue

            # Handle file submissions
            for file in submission['files']:
                file_type = file.get('type', self._determine_file_type(file['extension']))

                if file_type == 'code':
                    categorized['code'].append({**submission, 'current_file': file})
                elif file_type == 'text':
                    categorized['text'].append({**submission, 'current_file': file})
                else:
                    categorized['other'].append({**submission, 'current_file': file})

        return categorized

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)