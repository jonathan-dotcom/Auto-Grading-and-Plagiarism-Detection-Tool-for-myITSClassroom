import os
import time
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import json

from core.file_processor import FileProcessor
from core.grader import TextGrader, ShortAnswerGrader, CodeGrader, GradingManager
from core.plagiarism import TextPlagiarismDetector, CodePlagiarismDetector, PlagiarismManager
from core.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="MyITS Auto-Grader & Plagiarism Detector",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'grade'

if 'file_processor' not in st.session_state:
    st.session_state.file_processor = None

if 'submissions' not in st.session_state:
    st.session_state.submissions = None

if 'categorized_submissions' not in st.session_state:
    st.session_state.categorized_submissions = None

if 'grading_results' not in st.session_state:
    st.session_state.grading_results = []

if 'essay_grading_results' not in st.session_state:
    st.session_state.essay_grading_results = []
if 'short_answer_grading_results' not in st.session_state:
    st.session_state.short_answer_grading_results = []
if 'code_grading_results' not in st.session_state:
    st.session_state.code_grading_results = []

if 'plagiarism_results' not in st.session_state:
    st.session_state.plagiarism_results = {}

if 'report_paths' not in st.session_state:
    st.session_state.report_paths = {}

# C/C++ support - template state variables
if 'test_cases_template' not in st.session_state:
    st.session_state.test_cases_template = ""

if 'reference_code_template' not in st.session_state:
    st.session_state.reference_code_template = ""

# Create templates directory and C/C++ examples file on first run
if not os.path.exists('templates'):
    os.makedirs('templates', exist_ok=True)

if not os.path.exists('templates/c_cpp_examples.json'):
    with open('templates/c_cpp_examples.json', 'w') as f:
        json.dump({
            "c_basic_test_cases": [
                {
                    "function_name": "add",
                    "input": "2, 3",
                    "expected_output": "5"
                },
                {
                    "function_name": "subtract",
                    "input": "5, 3",
                    "expected_output": "2"
                },
                {
                    "assertion": "add(2, 3) == 5 && subtract(5, 3) == 2"
                }
            ],
            "cpp_basic_test_cases": [
                {
                    "function_name": "add",
                    "input": "2, 3",
                    "expected_output": "5"
                },
                {
                    "function_name": "multiply",
                    "input": "2, 3",
                    "expected_output": "6"
                },
                {
                    "assertion": "factorial(5) == 120"
                }
            ],
            "c_advanced_test_cases": [
                {
                    "function_name": "findMax",
                    "input": "(int[]){1, 5, 3, 9, 2}, 5",
                    "expected_output": "9"
                },
                {
                    "function_name": "isPrime",
                    "input": "17",
                    "expected_output": "1"
                },
                {
                    "assertion": "strcmp(reverseString(\"hello\"), \"olleh\") == 0"
                }
            ],
            "cpp_advanced_test_cases": [
                {
                    "function_name": "findLargest",
                    "input": "std::vector<int>{1, 5, 3, 9, 2}",
                    "expected_output": "9"
                },
                {
                    "function_name": "countVowels",
                    "input": "\"Hello World\"",
                    "expected_output": "3"
                },
                {
                    "assertion": "isPalindrome(\"racecar\") == true && isPalindrome(\"hello\") == false"
                }
            ]
        }, f, indent=2)

# Sidebar navigation
st.sidebar.title("Navigation")

pages = {
    'grade': 'Auto-Grading',
    'plagiarism': 'Plagiarism Detection',
    'report': 'Generate Reports'
}

for page_id, page_name in pages.items():
    if st.sidebar.button(page_name):
        st.session_state.current_page = page_id
        st.rerun()

    # Show current page name in sidebar
st.sidebar.markdown(f"**Current Page:** {pages[st.session_state.current_page]}")

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Auto-Grading & Plagiarism Detection Tool for myITS Classroom")
st.sidebar.markdown("v1.0.0")


# Reusable File Upload Section
def render_file_upload_section():
    st.header("Upload Assignment Files")
    st.markdown("Please upload the myITS Classroom assignment export ZIP file to begin.")

    uploaded_file = st.file_uploader("Upload myITS Classroom assignment export ZIP", type=['zip'],
                                     key="file_uploader_main")

    if uploaded_file is not None:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            zip_path = tmp_file.name

        with st.spinner("Processing ZIP file..."):
            try:
                # Process the zip file
                file_processor = FileProcessor()
                extract_dir = file_processor.extract_zip(zip_path)
                submissions = file_processor.parse_moodle_structure(extract_dir)

                # Store in session state
                st.session_state.file_processor = file_processor
                st.session_state.submissions = submissions
                st.session_state.categorized_submissions = file_processor.categorize_submissions(submissions)

                # Show success message
                st.success(f"Successfully processed ZIP file! Found {len(submissions)} student submissions.")

                # Display submission stats
                st.subheader("Submission Overview")
                st.markdown(f"**Total Submissions:** {len(submissions)}")

                categorized = st.session_state.categorized_submissions

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Code Files", len(categorized['code']))
                with col2:
                    st.metric("Text Files", len(categorized['text']))
                with col3:
                    st.metric("Other Files", len(categorized['other']))

                # Prepare data for chart
                file_types = {}
                for submission in submissions:
                    for file_info in submission['files']:  # Ensure this matches your structure
                        ext = file_info['extension']
                        if ext in file_types:
                            file_types[ext] += 1
                        else:
                            file_types[ext] = 1

                chart_data = pd.DataFrame({
                    'Extension': list(file_types.keys()),
                    'Count': list(file_types.values())
                })

                if not chart_data.empty:
                    st.subheader("File Extensions")
                    fig = px.bar(chart_data, x='Extension', y='Count', color='Extension')
                    st.plotly_chart(fig)

                st.subheader("Student Submissions")
                student_data = []
                for submission in submissions:
                    student_data.append({
                        'Student ID': submission['student_id'],
                        'Student Name': submission['student_name'],
                        'Files': len(submission['files']),
                        'Type': submission.get('submission_type', 'file')  # Assuming 'submission_type' exists
                    })
                student_df = pd.DataFrame(student_data)
                st.dataframe(student_df)

                st.info("Files uploaded and processed. You can now proceed with the actions on this page.")
                # Use a button to trigger rerun, giving user control
                if st.button("Continue"):
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")
                # Clean up session state if processing fails
                st.session_state.file_processor = None
                st.session_state.submissions = None
                st.session_state.categorized_submissions = None
            finally:
                # Clean up the temporary zip file
                if os.path.exists(zip_path):
                    os.unlink(zip_path)
        return True  # Indicates successful upload and processing trigger
    return False  # Indicates no file uploaded or processed yet


# Auto-Grading Page
def render_grading_page():
    st.title("Auto-Grading")

    if st.session_state.submissions is None or st.session_state.categorized_submissions is None:
        render_file_upload_section()
        return

    st.success(f"Submissions loaded: {len(st.session_state.submissions)} students. You can now configure grading.")
    st.markdown("---")

    # Assignment configuration
    st.header("Assignment Configuration")
    assignment_type = st.selectbox(
        "Assignment Type",
        options=["Essay", "Short Answer", "Code"],
        help="Select the type of assignment to apply appropriate grading criteria"
    )

    st.header("Grading Setup")

    if assignment_type == "Essay":
        st.subheader("Essay Grading")
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions suitable for essay grading.")

        if not text_submissions:
            st.warning("No text submissions found for essay grading.")
            return

        with st.form("essay_grading_form"):
            st.markdown("#### Grading Criteria")
            answer_key = st.text_area("Answer Key (Model Answer)", height=200, placeholder="Enter the model answer...")
            col1, col2 = st.columns(2)
            with col1:
                total_points = st.number_input("Total Points", min_value=1, value=100, key="essay_total_points")
            with col2:
                similarity_threshold = st.slider("Similarity Threshold for Full Points", 0.5, 1.0, 0.7, 0.05,
                                                 key="essay_sim_thresh")
            submitted = st.form_submit_button("Start Essay Grading")

            if submitted and answer_key:
                with st.spinner("Grading essay submissions..."):
                    grading_manager = GradingManager()
                    text_grader = TextGrader(answer_key=answer_key, total_points=total_points,
                                             threshold=similarity_threshold)
                    grading_manager.register_grader('text', text_grader)
                    results = []
                    for submission in text_submissions:
                        try:
                            file_path = submission['current_file']['path']
                            content = st.session_state.file_processor.get_file_content(file_path)
                            grade_result = grading_manager.grade_submission('text', content)
                            results.append({
                                'student_id': submission['student_id'],
                                'student_name': submission['student_name'],
                                'file_name': submission['current_file']['name'],
                                'grade': grade_result
                            })
                        except Exception as e:
                            st.error(f"Error grading {submission['student_name']}'s submission: {e}")
                    st.session_state.grading_results.extend(results)
                    st.session_state.essay_grading_results = results
                    st.success(f"Successfully graded {len(results)} essay submissions!")
                    st.rerun()

        if st.session_state.essay_grading_results:
            results_to_display = st.session_state.essay_grading_results
            result_data = [{
                'Student ID': r['student_id'], 'Student Name': r['student_name'],
                'Points': r['grade']['points'], 'Percentage': f"{r['grade']['percentage']}%",
                'Similarity': f"{r['grade']['similarity'] * 100:.2f}%"
            } for r in results_to_display]
            result_df = pd.DataFrame(result_data)
            st.dataframe(result_df)

            if results_to_display:  # Ensure there are results before calculating stats
                avg_grade = sum(r['grade']['points'] for r in results_to_display) / len(results_to_display)
                max_grade = max(r['grade']['points'] for r in results_to_display)
                min_grade = min(r['grade']['points'] for r in results_to_display)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Average Grade", f"{avg_grade:.2f}")
                with col2: st.metric("Highest Grade", f"{max_grade:.2f}")
                with col3: st.metric("Lowest Grade", f"{min_grade:.2f}")
                grades = [r['grade']['points'] for r in results_to_display]
                fig = px.histogram(x=grades, nbins=10, labels={'x': 'Points', 'y': 'Count'}, title='Grade Distribution')
                st.plotly_chart(fig)

    elif assignment_type == "Short Answer":
        st.subheader("Short Answer Grading")
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions suitable for short answer grading.")

        if not text_submissions:
            st.warning("No text submissions found for short answer grading.")
            return

        with st.form("short_answer_grading_form"):
            st.markdown("#### Grading Criteria")
            answer_key = st.text_area("Model Answer", height=100, placeholder="Enter the model/reference answer",
                                      key="sa_answer_key")
            keywords_text = st.text_area("Key Concepts (one per line)", height=100, placeholder="keyword1\nkeyword2",
                                         key="sa_keywords")
            col1, col2 = st.columns(2)
            with col1:
                total_points = st.number_input("Total Points", min_value=1, value=10, key="sa_total_points")
            with col2:
                keyword_weight = st.slider("Keyword Matching Weight (%)", 0, 100, 70, key="sa_keyword_weight")
            submitted = st.form_submit_button("Start Short Answer Grading")

            if submitted and answer_key and keywords_text:
                with st.spinner("Grading short answer submissions..."):
                    keyword_list = [k.strip() for k in keywords_text.split('\n') if k.strip()]
                    grading_manager = GradingManager()
                    short_answer_grader = ShortAnswerGrader(
                        answer_key=answer_key, keywords=keyword_list, total_points=total_points,
                        keyword_match_points=(keyword_weight / 100) * total_points,
                        similarity_points=((100 - keyword_weight) / 100) * total_points
                    )
                    grading_manager.register_grader('short_answer', short_answer_grader)
                    results = []
                    for submission in text_submissions:
                        try:
                            file_path = submission['current_file']['path']
                            content = st.session_state.file_processor.get_file_content(file_path)
                            grade_result = grading_manager.grade_submission('short_answer', content)
                            results.append({
                                'student_id': submission['student_id'],
                                'student_name': submission['student_name'],
                                'file_name': submission['current_file']['name'],
                                'grade': grade_result
                            })
                        except Exception as e:
                            st.error(f"Error grading {submission['student_name']}'s submission: {e}")
                    st.session_state.grading_results.extend(results)  # Append to general
                    st.session_state.short_answer_grading_results = results  # Specific
                    st.success(f"Successfully graded {len(results)} short answer submissions!")
                    st.rerun()

        if st.session_state.short_answer_grading_results:
            results_to_display = st.session_state.short_answer_grading_results
            result_data = [{
                'Student ID': r['student_id'], 'Student Name': r['student_name'],
                'Points': r['grade']['points'], 'Percentage': f"{r['grade']['percentage']}%",
                'Keywords Matched': f"{len(r['grade']['matched_keywords'])}/{len(r['grade']['matched_keywords']) + len(r['grade']['missing_keywords'])}"
            } for r in results_to_display]
            result_df = pd.DataFrame(result_data)
            st.dataframe(result_df)

            if results_to_display:
                st.subheader("Detailed Grading")
                student_names = [r['student_name'] for r in results_to_display]
                if not student_names:
                    st.info("No student results to display details for.")
                else:
                    selected_student = st.selectbox(
                        "Select student to view detailed feedback",
                        options=student_names,
                        key="short_answer_student_selector"
                    )
                    selected_result = next((r for r in results_to_display if r['student_name'] == selected_student),
                                           None)
                    if selected_result:
                        st.markdown(f"**Student:** {selected_result['student_name']}")
                        st.markdown(
                            f"**Points:** {selected_result['grade']['points']}/{selected_result['grade']['max_points']}")
                        st.markdown("**Matched Keywords:**")
                        st.write(", ".join(selected_result['grade']['matched_keywords']) if selected_result['grade'][
                            'matched_keywords'] else "None")
                        st.markdown("**Missing Keywords:**")
                        st.write(", ".join(selected_result['grade']['missing_keywords']) if selected_result['grade'][
                            'missing_keywords'] else "None")
                        st.markdown(f"**Overall Similarity:** {selected_result['grade']['similarity'] * 100:.2f}%")
                        st.markdown("**Feedback:**");
                        st.write(selected_result['grade']['feedback'])


    elif assignment_type == "Code":
        st.subheader("Code Grading")
        code_submissions = st.session_state.categorized_submissions['code']
        st.info(f"Found {len(code_submissions)} code submissions suitable for code grading.")

        if not code_submissions:
            st.warning("No code submissions found for grading.")
            return

        # Updated language selection with C/C++ support
        language = st.selectbox(
            "Programming Language",
            ["python", "c", "c++", "java", "javascript"],
            index=0,
            key="code_lang"
        )

        # C/C++ specific settings - MOVED OUTSIDE THE FORM
        if language == "c" or language == "c++":
            st.subheader("Language-specific Settings")

            # Load example test cases
            try:
                with open(os.path.join(os.path.dirname(__file__), 'templates', 'c_cpp_examples.json'), 'r') as f:
                    test_examples = json.load(f)

                example_key = f"{'c' if language == 'c' else 'cpp'}_basic_test_cases"
                advanced_key = f"{'c' if language == 'c' else 'cpp'}_advanced_test_cases"

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Use Basic {language.upper()} Test Template"):
                        st.session_state.test_cases_template = json.dumps(test_examples.get(example_key, []), indent=2)
                        st.rerun()  # Add rerun to update the UI
                with col2:
                    if st.button(f"Use Advanced {language.upper()} Test Template"):
                        st.session_state.test_cases_template = json.dumps(test_examples.get(advanced_key, []), indent=2)
                        st.rerun()  # Add rerun to update the UI
            except Exception as e:
                st.warning(f"Could not load example templates: {e}")

            # Provide example reference implementation
            example_ref_code = ""
            if language == "c":
                example_ref_code = """// Example C reference implementation
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

// For advanced test case
int findMax(int arr[], int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}
"""
            else:  # C++
                example_ref_code = """// Example C++ reference implementation
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

// For advanced test case
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}

int findLargest(std::vector<int> numbers) {
    return *std::max_element(numbers.begin(), numbers.end());
}
"""

            if st.button(f"Use Example {language.upper()} Reference Code"):
                st.session_state.reference_code_template = example_ref_code
                st.rerun()  # Add rerun to update the UI

            # Show compilation environment information
            st.info(f"""
                **{language.upper()} Compilation Environment:**
                - Compiler: {'GCC' if language == 'c' else 'G++'}
                - Standard: {'C11' if language == 'c' else 'C++17'}
                - Flags: -Wall
                - Docker Image: gcc:latest
            """)

        # FORM SECTION
        with st.form("code_grading_form"):
            st.markdown("#### Grading Criteria")

            # Reference implementation text area (with template support)
            reference_code = st.text_area(
                "Reference Implementation (Optional)",
                height=200,
                value=st.session_state.get('reference_code_template', ''),
                placeholder="Enter reference code...",
                key="code_ref_impl"
            )

            st.markdown("#### Test Cases (JSON format)")
            st.code('''
                    [
                        { "function_name": "add", "input": "2,3", "expected_output": "5" },
                        { "assertion": "submission.multiply(2,3) == 6" }
                    ]
            ''')  # Provide a simpler, more generic example

            # Test cases text area (with template support)
            test_cases_json = st.text_area(
                "Test Cases (JSON)",
                height=200,
                value=st.session_state.get('test_cases_template', ''),
                placeholder="Enter test cases...",
                key="code_test_cases"
            )

            total_points = st.number_input("Total Points", min_value=1, value=100, key="code_total_points")
            submitted = st.form_submit_button("Start Code Grading")

        # PROCESSING SECTION - Outside the form
        if submitted:
            if not test_cases_json:
                st.error("Test cases JSON cannot be empty. Please provide test cases.")
            else:
                with st.spinner("Grading code submissions..."):
                    try:
                        test_cases = json.loads(test_cases_json)
                        grading_manager = GradingManager()
                        code_grader = CodeGrader(
                            answer_key=reference_code, test_cases=test_cases,
                            language=language, total_points=total_points
                        )
                        grading_manager.register_grader('code', code_grader)
                        results = []
                        for submission in code_submissions:
                            try:
                                file_path = submission['current_file']['path']
                                content = st.session_state.file_processor.get_file_content(file_path)
                                grade_result = grading_manager.grade_submission('code', content)
                                results.append({
                                    'student_id': submission['student_id'],
                                    'student_name': submission['student_name'],
                                    'file_name': submission['current_file']['name'],
                                    'grade': grade_result
                                })
                            except Exception as e:
                                st.error(f"Error grading {submission['student_name']}'s submission: {e}")
                        st.session_state.grading_results.extend(results)  # Append to general
                        st.session_state.code_grading_results = results  # Specific
                        st.success(f"Successfully graded {len(results)} code submissions!")
                        st.rerun()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for test cases. Please check your input.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during code grading: {e}")

        # RESULTS DISPLAY SECTION
        if st.session_state.code_grading_results:
            results_to_display = st.session_state.code_grading_results
            result_data = [{
                'Student ID': r['student_id'], 'Student Name': r['student_name'],
                'Points': r['grade']['points'], 'Percentage': f"{r['grade']['percentage']}%",
                'Passed Tests': f"{r['grade']['passed_tests']}/{r['grade']['total_tests']}"
            } for r in results_to_display]
            result_df = pd.DataFrame(result_data)
            st.dataframe(result_df)

            if results_to_display:
                st.subheader("Test Details")
                student_names = [r['student_name'] for r in results_to_display]
                if not student_names:
                    st.info("No student results to display details for.")
                else:
                    selected_student_code = st.selectbox(
                        "Select student to view test results",
                        options=student_names,
                        key="code_student_selector"
                    )
                    selected_result_code = next(
                        (r for r in results_to_display if r['student_name'] == selected_student_code), None)
                    if selected_result_code:
                        st.markdown(f"**Student:** {selected_result_code['student_name']}")
                        st.markdown(
                            f"**Points:** {selected_result_code['grade']['points']}/{selected_result_code['grade']['max_points']}")
                        st.markdown("**Test Results:**")
                        for test in selected_result_code['grade']['test_results']:
                            if test['passed']:
                                st.success(f"✅ {test['name']} - Passed")
                            else:
                                st.error(f"❌ {test['name']} - Failed: {test['message']}")

                avg_grade = sum(r['grade']['points'] for r in results_to_display) / len(results_to_display)
                max_grade = max(r['grade']['points'] for r in results_to_display)
                min_grade = min(r['grade']['points'] for r in results_to_display)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Grade", f"{avg_grade:.2f}")
                with col2:
                    st.metric("Highest Grade", f"{max_grade:.2f}")
                with col3:
                    st.metric("Lowest Grade", f"{min_grade:.2f}")
                grades = [r['grade']['points'] for r in results_to_display]
                fig = px.histogram(x=grades, nbins=10, labels={'x': 'Points', 'y': 'Count'}, title='Grade Distribution')
                st.plotly_chart(fig)

    st.markdown("---")
    st.markdown(
        "**Next Steps:** Move to the Plagiarism Detection tab to check for similarities, or Generate Reports if grading is complete.")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("Proceed to Plagiarism Detection", key="nav_to_plag_from_grade"):
            st.session_state.current_page = 'plagiarism'
            st.rerun()
    with col_nav2:
        if st.button("Proceed to Generate Reports", key="nav_to_report_from_grade"):
            st.session_state.current_page = 'report'
            st.rerun()


# Plagiarism Detection Page
def render_plagiarism_page():
    st.title("Plagiarism Detection")

    # Check if we have submissions, if not, render upload section
    if st.session_state.submissions is None or st.session_state.categorized_submissions is None:
        render_file_upload_section()
        return  # Stop further rendering until files are uploaded

    st.success(
        f"Submissions loaded: {len(st.session_state.submissions)} students. You can now configure plagiarism detection.")
    st.markdown("---")

    st.header("1. Select Submission Type for Plagiarism Detection")
    tab1, tab2 = st.tabs(["Text/Essay", "Code"])

    with tab1:
        st.subheader("Text Plagiarism Detection")
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions.")

        if not text_submissions:
            st.warning("No text submissions available for plagiarism detection.")
        else:
            with st.form("text_plagiarism_form"):
                st.markdown("#### Detection Settings")
                similarity_threshold_text = st.slider(
                    "Similarity Threshold to Flag Plagiarism", 0.5, 1.0, 0.8, 0.05,
                    help="Minimum similarity to flag (0.8 recommended)", key="text_plag_thresh"
                )
                submitted_text_plag = st.form_submit_button("Start Text Plagiarism Detection")

                if submitted_text_plag:
                    with st.spinner("Detecting plagiarism in text submissions..."):
                        plagiarism_manager = PlagiarismManager()
                        text_detector = TextPlagiarismDetector(threshold=similarity_threshold_text)
                        plagiarism_manager.register_detector('text', text_detector)
                        # Prepare submissions in the format expected by PlagiarismManager
                        formatted_submissions = []
                        for sub in text_submissions:
                            try:
                                content = st.session_state.file_processor.get_file_content(sub['current_file']['path'])
                                formatted_submissions.append({
                                    'student_id': sub['student_id'],
                                    'student_name': sub['student_name'],
                                    'file_name': sub['current_file']['name'],
                                    'content': content
                                })
                            except Exception as e:
                                st.warning(f"Could not read file for {sub['student_name']}: {e}")

                        if formatted_submissions:
                            report = plagiarism_manager.detect_plagiarism('text', formatted_submissions)
                            st.session_state.plagiarism_results['text'] = report
                            st.success(f"Successfully compared {len(formatted_submissions)} text submissions!")
                            st.rerun()
                        else:
                            st.error("No text content could be processed for plagiarism detection.")

            if 'text' in st.session_state.plagiarism_results:
                report = st.session_state.plagiarism_results['text']
                st.markdown(f"#### Plagiarism Summary (Text)")
                st.markdown(f"Total Submissions Compared: {report['total_submissions']}")
                st.markdown(f"Flagged Pairs: {report['flagged_pairs']}")
                st.markdown(
                    f"Unique Flagged Students: {len(report['flagged_student_ids'])}")  # Assuming your report has this
                st.markdown(f"Similarity Threshold: {report['threshold'] * 100}%")

                if report['flagged_pairs'] > 0 and report['results']:
                    st.markdown("#### Flagged Text Submissions")
                    flagged_data = [{
                        'Student 1': f"{r['student1_name']} ({r['student1_id']})",
                        'Student 2': f"{r['student2_name']} ({r['student2_id']})",
                        'Similarity': f"{r['similarity'] * 100:.2f}%",
                        'File 1': r['student1_file'], 'File 2': r['student2_file']
                    } for r in report['results'] if r['flagged']]  # Make sure to filter only flagged

                    if flagged_data:
                        flagged_df = pd.DataFrame(flagged_data)
                        st.dataframe(flagged_df)

                        # Heatmap
                        all_students_involved = set()
                        for r in report['results']:
                            all_students_involved.add(r['student1_name'])
                            all_students_involved.add(r['student2_name'])

                        student_list = sorted(list(all_students_involved))
                        if len(student_list) > 1:  # Heatmap needs at least 2 students
                            heatmap_data = pd.DataFrame(0.0, index=student_list, columns=student_list)
                            for r in report['results']:
                                heatmap_data.loc[r['student1_name'], r['student2_name']] = r['similarity'] * 100
                                heatmap_data.loc[r['student2_name'], r['student1_name']] = r[
                                                                                               'similarity'] * 100  # Symmetric

                            fig = px.imshow(heatmap_data, labels=dict(x="Student", y="Student", color="Similarity %"),
                                            x=heatmap_data.columns, y=heatmap_data.index,
                                            color_continuous_scale='Blues')
                            st.plotly_chart(fig)
                        elif flagged_data:  # Only one pair or no pairs to form a matrix
                            st.info("Not enough unique student pairs to generate a heatmap for text submissions.")

                elif report['results'] is not None:  # results exist but no flagged pairs
                    st.success("No text plagiarism detected above the threshold!")

    with tab2:
        st.subheader("Code Plagiarism Detection")
        code_submissions = st.session_state.categorized_submissions['code']
        st.info(f"Found {len(code_submissions)} code submissions.")

        if not code_submissions:
            st.warning("No code submissions available for plagiarism detection.")
        else:
            # Language selection for plagiarism detection (moved outside the form)
            language_code_plag = st.selectbox(
                "Programming Language",
                ["python", "c", "c++", "java", "javascript"],
                index=0,
                key="code_plag_lang"
            )

            # Add information about C/C++ plagiarism detection
            if language_code_plag in ["c", "c++"]:
                st.info(f"""
                    **{language_code_plag.upper()} Plagiarism Detection:**
                    - Token Analysis: Identifies similarity in code tokens after removing comments and whitespace
                    - Structure Analysis: Compares function organization, loops, conditionals, and complexity
                    - {'Class & Template Analysis: Detects similarities in class structures and template usage' if language_code_plag == 'c++' else ''}
                """)

            with st.form("code_plagiarism_form"):
                st.markdown("#### Detection Settings")
                similarity_threshold_code = st.slider(
                    "Similarity Threshold to Flag Plagiarism", 0.5, 1.0, 0.8, 0.05,
                    help="Minimum similarity to flag (0.8 recommended)", key="code_plag_thresh"
                )
                submitted_code_plag = st.form_submit_button("Start Code Plagiarism Detection")

                if submitted_code_plag:
                    with st.spinner("Detecting plagiarism in code submissions..."):
                        plagiarism_manager = PlagiarismManager()
                        code_detector = CodePlagiarismDetector(threshold=similarity_threshold_code,
                                                               language=language_code_plag)
                        plagiarism_manager.register_detector('code', code_detector)

                        formatted_submissions_code = []
                        for sub in code_submissions:
                            try:
                                content = st.session_state.file_processor.get_file_content(sub['current_file']['path'])
                                formatted_submissions_code.append({
                                    'student_id': sub['student_id'],
                                    'student_name': sub['student_name'],
                                    'file_name': sub['current_file']['name'],
                                    'content': content
                                })
                            except Exception as e:
                                st.warning(f"Could not read code file for {sub['student_name']}: {e}")

                        if formatted_submissions_code:
                            report_code = plagiarism_manager.detect_plagiarism('code', formatted_submissions_code)
                            st.session_state.plagiarism_results['code'] = report_code
                            st.success(f"Successfully compared {len(formatted_submissions_code)} code submissions!")
                            st.rerun()
                        else:
                            st.error("No code content could be processed for plagiarism detection.")

            if 'code' in st.session_state.plagiarism_results:
                report_code = st.session_state.plagiarism_results['code']
                st.markdown(f"#### Plagiarism Summary (Code)")
                st.markdown(f"Total Submissions Compared: {report_code['total_submissions']}")
                st.markdown(f"Flagged Pairs: {report_code['flagged_pairs']}")
                st.markdown(f"Unique Flagged Students: {len(report_code['flagged_student_ids'])}")
                st.markdown(f"Similarity Threshold: {report_code['threshold'] * 100}%")

                if report_code['flagged_pairs'] > 0 and report_code['results']:
                    st.markdown("#### Flagged Code Submissions")
                    flagged_data_code = [{
                        'Student 1': f"{r['student1_name']} ({r['student1_id']})",
                        'Student 2': f"{r['student2_name']} ({r['student2_id']})",
                        'Overall Sim.': f"{r['similarity'] * 100:.2f}%",  # Assuming 'similarity' is overall
                        'Token Sim.': f"{r.get('token_similarity', 0) * 100:.2f}%",  # Use .get for safety
                        'AST/Struct Sim.': f"{r.get('ast_similarity', r.get('struct_similarity', 0)) * 100:.2f}%"
                    } for r in report_code['results'] if r['flagged']]

                    if flagged_data_code:
                        flagged_df_code = pd.DataFrame(flagged_data_code)
                        st.dataframe(flagged_df_code)

                        # Heatmap for code
                        all_students_code = set()
                        for r in report_code['results']:
                            all_students_code.add(r['student1_name'])
                            all_students_code.add(r['student2_name'])

                        student_list_code = sorted(list(all_students_code))
                        if len(student_list_code) > 1:
                            heatmap_data_code = pd.DataFrame(0.0, index=student_list_code, columns=student_list_code)
                            for r in report_code['results']:
                                heatmap_data_code.loc[r['student1_name'], r['student2_name']] = r['similarity'] * 100
                                heatmap_data_code.loc[r['student2_name'], r['student1_name']] = r['similarity'] * 100

                            fig_code = px.imshow(heatmap_data_code,
                                                 labels=dict(x="Student", y="Student", color="Similarity %"),
                                                 x=heatmap_data_code.columns, y=heatmap_data_code.index,
                                                 color_continuous_scale='Reds')
                            st.plotly_chart(fig_code)
                        elif flagged_data_code:
                            st.info("Not enough unique student pairs to generate a heatmap for code submissions.")
                elif report_code['results'] is not None:
                    st.success("No code plagiarism detected above the threshold!")

    st.markdown("---")
    st.markdown("**Next Steps:** Move to the Generate Reports tab to create CSV grade reports and plagiarism reports.")
    if st.button("Proceed to Generate Reports", key="nav_to_report_from_plag"):
        st.session_state.current_page = 'report'
        st.rerun()


def render_report_page():
    st.title("Generate Reports")
    if not st.session_state.grading_results and not st.session_state.plagiarism_results:
        st.warning("No grading or plagiarism results available. Please process submissions first.")
        col_nav_rep1, col_nav_rep2 = st.columns(2)
        with col_nav_rep1:
            if st.button("Go to Auto-Grading Page", key="nav_to_grade_from_report"):
                st.session_state.current_page = 'grade'
                st.rerun()
        with col_nav_rep2:
            if st.button("Go to Plagiarism Detection Page", key="nav_to_plag_from_report"):
                st.session_state.current_page = 'plagiarism'
                st.rerun()
        return

    st.header("1. Configure Reports")

    with st.form("report_form"):
        st.markdown("#### Report Settings")
        assignment_name = st.text_input("Assignment Name", value="Assignment", help="Name for report filenames")
        st.markdown("#### Select Reports to Generate")
        generate_grade_csv = st.checkbox("Generate Grades CSV (for myITS Classroom import)", value=True,
                                         disabled=not st.session_state.grading_results)
        generate_plagiarism_html = st.checkbox("Generate Plagiarism Report (HTML)", value=True,
                                               disabled=not st.session_state.plagiarism_results)
        generate_summary = st.checkbox("Generate Overall Summary Report (JSON)", value=True, disabled=not (
                    st.session_state.grading_results or st.session_state.plagiarism_results))

        submitted_reports = st.form_submit_button("Generate Reports")

        if submitted_reports:
            if not (generate_grade_csv or generate_plagiarism_html or generate_summary):
                st.warning("Please select at least one type of report to generate.")
            else:
                with st.spinner("Generating reports..."):
                    report_generator = ReportGenerator()
                    reports_generated_paths = {}  # Store paths of generated reports

                    # Ensure grading_results is a list of dictionaries with 'student_id', 'student_name', and 'grade' (dict with 'points')
                    valid_grading_results = [
                        res for res in st.session_state.grading_results
                        if isinstance(res, dict) and 'student_id' in res and 'grade' in res and isinstance(res['grade'],
                                                                                                           dict) and 'points' in
                           res['grade']
                    ]

                    if generate_grade_csv and valid_grading_results:
                        try:
                            grade_csv_path = report_generator.generate_grade_csv(
                                assignment_name,
                                valid_grading_results  # Use the consolidated list
                            )
                            reports_generated_paths['grade_csv'] = grade_csv_path
                        except Exception as e:
                            st.error(f"Error generating grade CSV: {e}")
                    elif generate_grade_csv:
                        st.warning("No valid grading results available to generate Grades CSV.")

                    if generate_plagiarism_html and st.session_state.plagiarism_results:
                        for report_type, report_content in st.session_state.plagiarism_results.items():
                            if report_content and report_content.get(
                                    'results'):  # Ensure content is not None and has results
                                try:
                                    html_path = report_generator.generate_plagiarism_html(
                                        f"{assignment_name}_{report_type}_plagiarism",  # More specific name
                                        report_content
                                    )
                                    reports_generated_paths[f'plagiarism_html_{report_type}'] = html_path
                                except Exception as e:
                                    st.error(f"Error generating plagiarism HTML for {report_type}: {e}")
                            else:
                                st.info(f"No plagiarism data to generate HTML report for {report_type}.")
                    elif generate_plagiarism_html:
                        st.warning("No plagiarism results available to generate HTML report.")

                    if generate_summary:
                        try:
                            summary_data_path = report_generator.generate_summary_report(
                                # Assuming it returns a path or data
                                assignment_name,
                                valid_grading_results,
                                st.session_state.plagiarism_results  # Pass the whole dict
                            )
                            reports_generated_paths['summary_json'] = summary_data_path  # Assuming JSON path or data
                        except Exception as e:
                            st.error(f"Error generating summary report: {e}")

                    st.session_state.report_paths = reports_generated_paths
                    if reports_generated_paths:
                        st.success(f"Successfully generated {len(reports_generated_paths)} reports!")
                    else:
                        st.info("No reports were generated based on selections and available data.")
                    st.rerun()  # To display download links outside the form

    # Display reports if available - outside the form
    if st.session_state.report_paths:
        st.header("2. Download Reports")
        for report_key, report_item in st.session_state.report_paths.items():
            if report_key == 'summary_json':  # If it's the summary data itself (not a path)
                st.subheader("Summary Report Data (JSON)")
                st.json(report_item)  # Assuming report_item is the JSON data
                # If generate_summary_report saves a file and returns a path:
                # report_name = os.path.basename(report_item)
                # with open(report_item, 'r') as f_json: # Read as text for JSON
                #     st.download_button(
                #         label=f"Download {report_name}", data=f_json.read(),
                #         file_name=report_name, mime='application/json'
                #     )
            elif os.path.exists(str(report_item)):  # Check if it's a valid file path
                report_path = str(report_item)
                report_name = os.path.basename(report_path)
                mime_type = 'text/csv' if report_name.endswith('.csv') else \
                    'text/html' if report_name.endswith('.html') else \
                        'application/json' if report_name.endswith('.json') else \
                            'application/octet-stream'
                try:
                    with open(report_path, 'rb') as f_report:
                        st.download_button(
                            label=f"Download {report_name}",
                            data=f_report,
                            file_name=report_name,
                            mime=mime_type
                        )
                except Exception as e:
                    st.error(f"Could not prepare {report_name} for download: {e}")
            else:
                st.warning(
                    f"Report file for '{report_key}' not found at path: {report_item}. It might not have been generated.")


# Main page rendering logic
current_page = st.session_state.current_page

if current_page == 'grade':
    render_grading_page()
elif current_page == 'plagiarism':
    render_plagiarism_page()
elif current_page == 'report':
    render_report_page()