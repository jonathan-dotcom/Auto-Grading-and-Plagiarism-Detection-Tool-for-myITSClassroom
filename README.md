# Auto-Grading and Plagiarism Detection Tool for myITSClassroom
 This project aims to build an Auto-Grading & Plagiarism Detection Tool for myITSClassroom application that processes assignments from the export of assignments in myITS Classroom offline, provides automatic grading based on the answer key, and detects plagiarism with AI algorithms.

Project Structure:
```markdown
The-Tool/
│
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # For isolated code execution environment
│
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── file_processor.py   # ZIP file extraction and processing
│   ├── grader.py           # Assignment grading logic
│   ├── plagiarism.py       # Plagiarism detection
│   └── report_generator.py # CSV/HTML/PDF report generation
│
├── models/                 # Data models
│   ├── __init__.py
│   ├── assignment.py       # Assignment data structure
│   ├── student.py          # Student data structure
│   └── submission.py       # Submission data structure
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── nlp_utils.py        # NLP utility functions
│   ├── code_utils.py       # Code analysis utilities
│   └── docker_utils.py     # Docker interaction utilities
│
├── database/               # Database interaction
│   ├── __init__.py
│   └── db_manager.py       # SQLite database manager
│
└── ui/                     # UI components
    ├── __init__.py
    ├── pages/              # Different app pages
    │   ├── __init__.py
    │   ├── upload_page.py
    │   ├── grading_page.py
    │   ├── plagiarism_page.py
    │   └── report_page.py
    └── components/         # Reusable UI components
        ├── __init__.py
        └── widgets.py
```

Set up a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Make sure Docker is installed and running (required for code execution in the grader)

You can verify with ```docker --version```. If not installed, download from Docker's website

Start the Streamlit application:
```bash
streamlit run app.py
```