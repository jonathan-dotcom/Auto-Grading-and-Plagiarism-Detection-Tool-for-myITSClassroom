import os
import time
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import logging
from datetime import datetime

# Import logging configuration
from logging_config import setup_logging, get_logger, log_function_entry, log_function_exit, log_error_with_traceback

# Set up logging at the start
log_files = setup_logging(log_level='DEBUG')

# Get loggers
logger = get_logger('streamlit_app')
file_logger = get_logger('file_processing')
grading_logger = get_logger('grading')
plagiarism_logger = get_logger('plagiarism_detection')

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

# Log application startup
logger.info("Streamlit application starting")
logger.info(f"Session ID: {st.session_state.get('session_id', 'new')}")

# Create session ID if not exists
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"New session created: {st.session_state.session_id}")

# Display log file paths in sidebar
with st.sidebar:
    with st.expander("📋 Debug Information", expanded=False):
        st.write("**Log Files Created:**")
        for log_type, log_path in log_files.items():
            st.write(f"• {log_type}: `{log_path}`")
        st.write("**Current Session:**")
        st.write(f"• ID: `{st.session_state.session_id}`")
        st.write(f"• Started: `{datetime.now().strftime('%H:%M:%S')}`")

# Language mappings
LANGUAGE_MAPPINGS = {
    'python': {'extensions': ['.py'], 'display': 'Python'},
    'c': {'extensions': ['.c'], 'display': 'C'},
    'cpp': {'extensions': ['.cpp', '.cc', '.cxx', '.c++'], 'display': 'C++'},
    'javascript': {'extensions': ['.js'], 'display': 'JavaScript'}
}

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

st.sidebar.markdown(f"**Current Page:** {pages[st.session_state.current_page]}")

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### Supported Languages")
st.sidebar.markdown("**Code Grading & Plagiarism:**")
st.sidebar.markdown("- Python (.py)")
st.sidebar.markdown("- C (.c)")
st.sidebar.markdown("- C++ (.cpp, .cc, .cxx)")
st.sidebar.markdown("- JavaScript (.js)")
st.sidebar.markdown("**Text Analysis:**")
st.sidebar.markdown("- Essays & Short Answers")
st.sidebar.markdown("- HTML submissions")
st.sidebar.markdown("- Various document formats")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Auto-Grading & Plagiarism Detection Tool")
st.sidebar.markdown("v2.0.0 - Multi-Language Support")


def get_language_from_extension(extension):
    """Get programming language from file extension."""
    for lang, info in LANGUAGE_MAPPINGS.items():
        if extension.lower() in info['extensions']:
            return lang
    return None


# Reusable File Upload Section
def render_file_upload_section():
    logger.info("Rendering file upload section")
    st.header("Upload Assignment Files")
    st.markdown("Please upload the myITS Classroom assignment export ZIP file to begin.")

    uploaded_file = st.file_uploader("Upload myITS Classroom assignment export ZIP", type=['zip'],
                                     key="file_uploader_main")

    if uploaded_file is not None:
        start_time = time.time()
        logger.info(f"File uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            zip_path = tmp_file.name
            logger.debug(f"Temporary file created: {zip_path}")

        with st.spinner("Processing ZIP file..."):
            try:
                logger.info("Starting ZIP file processing")

                # Process the zip file
                file_processor = FileProcessor()
                logger.debug("FileProcessor instance created")

                extract_dir = file_processor.extract_zip(zip_path)
                logger.info(f"ZIP extracted to: {extract_dir}")

                submissions = file_processor.parse_moodle_structure(extract_dir)
                logger.info(f"Found {len(submissions)} submissions after parsing")

                # Log submission details
                for i, submission in enumerate(submissions):
                    logger.debug(
                        f"Submission {i + 1}: {submission['student_name']} ({submission['student_id']}) - {len(submission['files'])} files")
                    for file_info in submission['files']:
                        logger.debug(
                            f"  File: {file_info['name']} ({file_info['extension']}) - {file_info.get('size', 0)} bytes")

                # Store in session state
                st.session_state.file_processor = file_processor
                st.session_state.submissions = submissions
                st.session_state.categorized_submissions = file_processor.categorize_submissions(submissions)

                processing_time = time.time() - start_time
                logger.info(f"File processing completed in {processing_time:.2f} seconds")

                # Show success message
                st.success(f"Successfully processed ZIP file! Found {len(submissions)} student submissions.")

                # Display submission stats
                st.subheader("Submission Overview")
                st.markdown(f"**Total Submissions:** {len(submissions)}")

                categorized = st.session_state.categorized_submissions
                logger.info(
                    f"Categorized submissions - Code: {len(categorized['code'])}, Text: {len(categorized['text'])}, Other: {len(categorized['other'])}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Code Files", len(categorized['code']))
                with col2:
                    st.metric("Text Files", len(categorized['text']))
                with col3:
                    st.metric("Other Files", len(categorized['other']))

                # Language distribution for code files
                if categorized['code']:
                    language_counts = {}
                    for submission in categorized['code']:
                        ext = submission['current_file']['extension']
                        lang = get_language_from_extension(ext)
                        if lang:
                            lang_display = LANGUAGE_MAPPINGS[lang]['display']
                            language_counts[lang_display] = language_counts.get(lang_display, 0) + 1

                    logger.info(f"Language distribution: {language_counts}")

                    if language_counts:
                        st.subheader("Programming Languages Detected")
                        lang_df = pd.DataFrame(list(language_counts.items()), columns=['Language', 'Count'])
                        fig = px.pie(lang_df, values='Count', names='Language', title='Code Submission Distribution')
                        st.plotly_chart(fig)

                # File extension chart
                file_types = {}
                for submission in submissions:
                    for file_info in submission['files']:
                        ext = file_info['extension']
                        file_types[ext] = file_types.get(ext, 0) + 1

                logger.debug(f"File extension distribution: {file_types}")

                if file_types:
                    st.subheader("File Extensions")
                    chart_data = pd.DataFrame({
                        'Extension': list(file_types.keys()),
                        'Count': list(file_types.values())
                    })
                    fig = px.bar(chart_data, x='Extension', y='Count', color='Extension')
                    st.plotly_chart(fig)

                # Student submissions table
                st.subheader("Student Submissions")
                student_data = []
                for submission in submissions:
                    student_data.append({
                        'Student ID': submission['student_id'],
                        'Student Name': submission['student_name'],
                        'Files': len(submission['files']),
                        'Type': submission.get('submission_type', 'file')
                    })
                student_df = pd.DataFrame(student_data)
                st.dataframe(student_df)

                st.info("Files uploaded and processed. You can now proceed with grading and plagiarism detection.")
                if st.button("Continue"):
                    st.rerun()

            except Exception as e:
                error_msg = f"An error occurred during file processing: {e}"
                logger.error(f"File processing error: {str(e)}")
                log_error_with_traceback(logger, e, "file upload processing")
                st.error(error_msg)
                # Clean up session state if processing fails
                st.session_state.file_processor = None
                st.session_state.submissions = None
                st.session_state.categorized_submissions = None
            finally:
                # Clean up the temporary zip file
                if os.path.exists(zip_path):
                    os.unlink(zip_path)
                    logger.debug(f"Cleaned up temporary file: {zip_path}")
        return True
    return False


# Auto-Grading Page
def render_grading_page():
    st.title("🎯 Auto-Grading")

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
        render_essay_grading()
    elif assignment_type == "Short Answer":
        render_short_answer_grading()
    elif assignment_type == "Code":
        render_code_grading()

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


def render_essay_grading():
    st.subheader("📝 Essay Grading")
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
        display_grading_results(st.session_state.essay_grading_results, "Essay")


def render_short_answer_grading():
    st.subheader("💭 Short Answer Grading")
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
                st.session_state.grading_results.extend(results)
                st.session_state.short_answer_grading_results = results
                st.success(f"Successfully graded {len(results)} short answer submissions!")
                st.rerun()

    if st.session_state.short_answer_grading_results:
        display_short_answer_results(st.session_state.short_answer_grading_results)


def render_code_grading():
    st.subheader("💻 Code Grading")
    code_submissions = st.session_state.categorized_submissions['code']
    st.info(f"Found {len(code_submissions)} code submissions suitable for code grading.")

    if not code_submissions:
        st.warning("No code submissions found for grading.")
        return

    # Detect languages in submissions
    detected_languages = set()
    for submission in code_submissions:
        ext = submission['current_file']['extension']
        lang = get_language_from_extension(ext)
        if lang:
            detected_languages.add(lang)

    if detected_languages:
        # Added safety check to prevent KeyError if a language is detected but not in mappings
        display_langs = [LANGUAGE_MAPPINGS[lang]['display'] for lang in detected_languages if lang in LANGUAGE_MAPPINGS]
        st.info(f"Detected languages: {', '.join(display_langs)}")

    with st.form("code_grading_form"):
        st.markdown("#### Grading Criteria")

        col1, col2 = st.columns(2)
        with col1:
            available_languages = ['python', 'c', 'cpp', 'javascript']
            language = st.selectbox("Programming Language",
                                    options=available_languages,
                                    format_func=lambda x: LANGUAGE_MAPPINGS[x]['display'],
                                    index=0, key="code_lang")
        with col2:
            total_points = st.number_input("Total Points", min_value=1, value=100, key="code_total_points")

        # FIX: Changed label and help text to indicate this field is optional.
        reference_code = st.text_area(
            "Reference Code (Optional)", 
            height=200,
            placeholder="Enter the model/reference code solution...",
            key="code_reference_solution",
            help="Provide a model solution to enable structural or similarity-based grading. If omitted, grading will be based solely on test case results."
        )


        st.markdown("#### Test Cases Configuration")

        # Language-specific test case examples
        if language == 'python':
            example_tests = '''[
    {"function_name": "add", "input": "2, 3", "expected_output": "5"},
    {"assertion": "submission.multiply(4, 5) == 20"},
    {"function_name": "factorial", "input": "5", "expected_output": "120"}
]'''
        elif language in ['c', 'cpp']:
            example_tests = '''[
    {"function_call": "add(2, 3)", "expected_output": "5"},
    {"function_call": "multiply(4, 5)", "expected_output": "20"}
]'''
        elif language == 'javascript':
            example_tests = '''[
    {"function_call": "submission.add(2, 3)", "expected_output": "5"},
    {"function_call": "submission.multiply(4, 5)", "expected_output": "20"}
]'''
        else:
            example_tests = '''[
    {"function_name": "function_name", "input": "input_values", "expected_output": "expected_result"}
]'''

        st.code(example_tests, language='json')
        test_cases_json = st.text_area("Test Cases (JSON)", height=200, placeholder="Enter test cases...",
                                       key="code_test_cases")

        submitted = st.form_submit_button("Start Code Grading")

    if submitted:
        # FIX: Removed the validation check for reference_code. Now, only test cases are required.
        if not test_cases_json:
            logger.warning("Code grading attempted with empty test cases")
            st.error("Test cases JSON cannot be empty. Please provide test cases.")
        else:
            start_time = time.time()
            logger.info(f"Starting code grading - Language: {language}, Students: {len(code_submissions)}")

            with st.spinner("Grading code submissions..."):
                try:
                    test_cases = json.loads(test_cases_json)
                    logger.info(f"Parsed {len(test_cases)} test cases")
                    logger.debug(f"Test cases: {test_cases}")

                    grading_manager = GradingManager()
                    
                    # FIX: Pass reference_code (which can be an empty string) to the grader.
                    # Your CodeGrader class should handle an empty or None answer_key gracefully.
                    code_grader = CodeGrader(
                        answer_key=reference_code, # This is now optional
                        test_cases=test_cases,
                        language=language, 
                        total_points=total_points
                    )
                    logger.debug(f"CodeGrader created for {language}")

                    grading_manager.register_grader('code', code_grader)
                    results = []

                    for i, submission in enumerate(code_submissions):
                        student_start = time.time()
                        student_name = submission['student_name']
                        logger.info(f"Grading {student_name} ({i + 1}/{len(code_submissions)})")

                        try:
                            file_path = submission['current_file']['path']
                            logger.debug(f"Reading file: {file_path}")

                            content = st.session_state.file_processor.get_file_content(file_path)
                            logger.debug(f"File content length: {len(content)} characters")
                            logger.debug(f"File content preview: {content[:200]}...")

                            grade_result = grading_manager.grade_submission('code', content)
                            student_time = time.time() - student_start

                            logger.info(
                                f"Graded {student_name} in {student_time:.2f}s - Score: {grade_result['points']}/{grade_result['max_points']}")
                            logger.debug(f"Grade result for {student_name}: {grade_result}")

                            results.append({
                                'student_id': submission['student_id'],
                                'student_name': submission['student_name'],
                                'file_name': submission['current_file']['name'],
                                'grade': grade_result
                            })

                        except Exception as e:
                            error_msg = f"Error grading {student_name}'s submission: {e}"
                            logger.error(error_msg)
                            log_error_with_traceback(grading_logger, e, f"grading {student_name}")
                            st.error(error_msg)

                    total_time = time.time() - start_time
                    logger.info(f"Code grading completed in {total_time:.2f}s - {len(results)} students graded")

                    st.session_state.grading_results.extend(results)
                    st.session_state.code_grading_results = results
                    st.success(f"Successfully graded {len(results)} code submissions!")
                    st.rerun()

                except json.JSONDecodeError as e:
                    error_msg = "Invalid JSON format for test cases. Please check your input."
                    logger.error(f"JSON decode error: {e}")
                    st.error(error_msg)
                except Exception as e:
                    error_msg = f"An unexpected error occurred during code grading: {e}"
                    logger.error(error_msg)
                    log_error_with_traceback(grading_logger, e, "code grading")
                    st.error(error_msg)

    if st.session_state.code_grading_results:
        display_code_results(st.session_state.code_grading_results)


def display_grading_results(results, assignment_type):
    """Display grading results with statistics."""
    result_data = [{
        'Student ID': r['student_id'],
        'Student Name': r['student_name'],
        'Points': r['grade']['points'],
        'Percentage': f"{r['grade']['percentage']}%",
        'Similarity': f"{r['grade']['similarity'] * 100:.2f}%" if 'similarity' in r['grade'] else 'N/A'
    } for r in results]
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df)

    if results:
        avg_grade = sum(r['grade']['points'] for r in results) / len(results)
        max_grade = max(r['grade']['points'] for r in results)
        min_grade = min(r['grade']['points'] for r in results)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Average Grade", f"{avg_grade:.2f}")
        with col2: st.metric("Highest Grade", f"{max_grade:.2f}")
        with col3: st.metric("Lowest Grade", f"{min_grade:.2f}")

        grades = [r['grade']['points'] for r in results]
        fig = px.histogram(x=grades, nbins=10, labels={'x': 'Points', 'y': 'Count'},
                           title=f'{assignment_type} Grade Distribution')
        st.plotly_chart(fig)


def display_short_answer_results(results):
    """Display short answer specific results."""
    result_data = [{
        'Student ID': r['student_id'],
        'Student Name': r['student_name'],
        'Points': r['grade']['points'],
        'Percentage': f"{r['grade']['percentage']}%",
        'Keywords Matched': f"{len(r['grade']['matched_keywords'])}/{len(r['grade']['matched_keywords']) + len(r['grade']['missing_keywords'])}"
    } for r in results]
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df)

    if results:
        st.subheader("Detailed Grading")
        student_names = [r['student_name'] for r in results]
        selected_student = st.selectbox("Select student to view detailed feedback", options=student_names,
                                        key="short_answer_student_selector")
        selected_result = next((r for r in results if r['student_name'] == selected_student), None)

        if selected_result:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Student:** {selected_result['student_name']}")
                st.markdown(
                    f"**Points:** {selected_result['grade']['points']}/{selected_result['grade']['max_points']}")
                st.markdown(f"**Overall Similarity:** {selected_result['grade']['similarity'] * 100:.2f}%")
            with col2:
                st.markdown("**Matched Keywords:**")
                st.write(", ".join(selected_result['grade']['matched_keywords']) if selected_result['grade'][
                    'matched_keywords'] else "None")
                st.markdown("**Missing Keywords:**")
                st.write(", ".join(selected_result['grade']['missing_keywords']) if selected_result['grade'][
                    'missing_keywords'] else "None")

            st.markdown("**Feedback:**")
            st.write(selected_result['grade']['feedback'])


def display_code_results(results):
    """Display code grading specific results."""
    result_data = [{
        'Student ID': r['student_id'],
        'Student Name': r['student_name'],
        'Points': r['grade']['points'],
        'Percentage': f"{r['grade']['percentage']}%",
        'Passed Tests': f"{r['grade']['passed_tests']}/{r['grade']['total_tests']}"
    } for r in results]
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df)

    if results:
        st.subheader("Test Details")
        student_names = [r['student_name'] for r in results]
        selected_student_code = st.selectbox("Select student to view test results", options=student_names,
                                             key="code_student_selector")
        selected_result_code = next((r for r in results if r['student_name'] == selected_student_code), None)

        if selected_result_code:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Student:** {selected_result_code['student_name']}")
                st.markdown(
                    f"**Points:** {selected_result_code['grade']['points']}/{selected_result_code['grade']['max_points']}")
                st.markdown(
                    f"**Tests Passed:** {selected_result_code['grade']['passed_tests']}/{selected_result_code['grade']['total_tests']}")
            with col2:
                st.markdown("**Test Results:**")
                for test in selected_result_code['grade']['test_results']:
                    if test['passed']:
                        st.success(f"✅ {test['name']} - Passed")
                    else:
                        st.error(f"❌ {test['name']} - Failed: {test['message']}")

        # Statistics
        avg_grade = sum(r['grade']['points'] for r in results) / len(results)
        max_grade = max(r['grade']['points'] for r in results)
        min_grade = min(r['grade']['points'] for r in results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Grade", f"{avg_grade:.2f}")
        with col2:
            st.metric("Highest Grade", f"{max_grade:.2f}")
        with col3:
            st.metric("Lowest Grade", f"{min_grade:.2f}")

        grades = [r['grade']['points'] for r in results]
        fig = px.histogram(x=grades, nbins=10, labels={'x': 'Points', 'y': 'Count'}, title='Code Grade Distribution')
        st.plotly_chart(fig)


# Plagiarism Detection Page
def render_plagiarism_page():
    st.title("🔍 Plagiarism Detection")

    if st.session_state.submissions is None or st.session_state.categorized_submissions is None:
        render_file_upload_section()
        return

    st.success(
        f"Submissions loaded: {len(st.session_state.submissions)} students. You can now configure plagiarism detection.")
    st.markdown("---")

    st.header("Select Submission Type for Plagiarism Detection")
    tab1, tab2 = st.tabs(["📝 Text/Essay", "💻 Code"])

    with tab1:
        render_text_plagiarism()

    with tab2:
        render_code_plagiarism()

    st.markdown("---")
    st.markdown("**Next Steps:** Move to the Generate Reports tab to create comprehensive reports.")
    if st.button("Proceed to Generate Reports", key="nav_to_report_from_plag"):
        st.session_state.current_page = 'report'
        st.rerun()


def render_text_plagiarism():
    st.subheader("Text Plagiarism Detection")
    text_submissions = st.session_state.categorized_submissions['text']
    st.info(f"Found {len(text_submissions)} text submissions.")

    if not text_submissions:
        st.warning("No text submissions available for plagiarism detection.")
        return

    with st.form("text_plagiarism_form"):
        st.markdown("#### Detection Settings")
        similarity_threshold_text = st.slider(
            "Similarity Threshold to Flag Plagiarism", 0.5, 1.0, 0.8, 0.05,
            help="Minimum similarity to flag as plagiarism (0.8 recommended)", key="text_plag_thresh"
        )
        st.info(
            " text plagiarism detection uses TF-IDF, fingerprinting, and n-gram analysis for improved accuracy.")
        submitted_text_plag = st.form_submit_button("Start Text Plagiarism Detection")

        if submitted_text_plag:
            start_time = time.time()
            logger.info("Starting text plagiarism detection")
            logger.info(f"Threshold: {similarity_threshold_text}, Submissions: {len(text_submissions)}")

            with st.spinner("Detecting plagiarism in text submissions..."):
                plagiarism_manager = PlagiarismManager()
                text_detector = TextPlagiarismDetector(threshold=similarity_threshold_text)
                plagiarism_manager.register_detector('text', text_detector)

                formatted_submissions = []
                for i, sub in enumerate(text_submissions):
                    logger.debug(f"Processing text submission {i + 1}: {sub['student_name']}")
                    try:
                        content = st.session_state.file_processor.get_file_content(sub['current_file']['path'])
                        logger.debug(f"Raw content length for {sub['student_name']}: {len(content)} chars")
                        logger.debug(f"Content preview: {content[:200]}...")

                        formatted_submissions.append({
                            'student_id': sub['student_id'],
                            'student_name': sub['student_name'],
                            'file_name': sub['current_file']['name'],
                            'content': content
                        })
                        logger.debug(f"Added {sub['student_name']} to formatted submissions")

                    except Exception as e:
                        error_msg = f"Could not read file for {sub['student_name']}: {e}"
                        logger.error(error_msg)
                        log_error_with_traceback(plagiarism_logger, e, f"reading file for {sub['student_name']}")
                        st.warning(error_msg)

                logger.info(
                    f"Successfully formatted {len(formatted_submissions)} text submissions for plagiarism detection")

                if formatted_submissions:
                    try:
                        logger.info("Starting plagiarism detection process")
                        report = plagiarism_manager.detect_plagiarism('text', formatted_submissions)
                        detection_time = time.time() - start_time

                        logger.info(f"Plagiarism detection completed in {detection_time:.2f}s")
                        logger.info(
                            f"Report summary: {report.get('total_comparisons', 0)} comparisons, {report.get('flagged_pairs', 0)} flagged pairs")
                        logger.debug(f"Full report: {report}")

                        st.session_state.plagiarism_results['text'] = report
                        st.success(f"Successfully compared {len(formatted_submissions)} text submissions!")
                        st.rerun()

                    except Exception as e:
                        error_msg = f"Error during plagiarism detection: {e}"
                        logger.error(error_msg)
                        log_error_with_traceback(plagiarism_logger, e, "text plagiarism detection")
                        st.error(error_msg)
                else:
                    error_msg = "No text content could be processed for plagiarism detection."
                    logger.error(error_msg)
                    st.error(error_msg)

    if 'text' in st.session_state.plagiarism_results:
        display_text_plagiarism_results(st.session_state.plagiarism_results['text'])


def render_code_plagiarism():
    st.subheader("Code Plagiarism Detection")
    code_submissions = st.session_state.categorized_submissions['code']
    st.info(f"Found {len(code_submissions)} code submissions.")

    if not code_submissions:
        st.warning("No code submissions available for plagiarism detection.")
        return

    # Detect available languages
    detected_languages = set()
    language_counts = {}
    for submission in code_submissions:
        ext = submission['current_file']['extension']
        lang = get_language_from_extension(ext)
        if lang:
            detected_languages.add(lang)
            lang_display = LANGUAGE_MAPPINGS[lang]['display']
            language_counts[lang_display] = language_counts.get(lang_display, 0) + 1

    if detected_languages:
        language_display_list = []
        for lang in detected_languages:
            display_name = LANGUAGE_MAPPINGS[lang]['display']
            file_count = language_counts.get(display_name, 0)
            language_display_list.append(f"{display_name} ({file_count} files)")
        st.info(f"Detected languages: {', '.join(language_display_list)}")

    with st.form("code_plagiarism_form"):
        st.markdown("#### Detection Settings")
        available_languages = ['python', 'c', 'cpp', 'javascript']
        language_code_plag = st.selectbox(
            "Programming Language",
            options=available_languages,
            format_func=lambda x: LANGUAGE_MAPPINGS[x]['display'],
            index=0, key="code_plag_lang"
        )
        similarity_threshold_code = st.slider(
            "Similarity Threshold to Flag Plagiarism", 0.5, 1.0, 0.8, 0.05,
            help="Minimum similarity to flag as plagiarism (0.8 recommended)", key="code_plag_thresh"
        )
        st.info(
            "Enhanced code plagiarism detection analyzes AST structure, token sequences, function signatures, and normalized code patterns.")
        submitted_code_plag = st.form_submit_button("Start Code Plagiarism Detection")

        if submitted_code_plag:
            with st.spinner("Detecting plagiarism in code submissions..."):
                plagiarism_manager = PlagiarismManager()
                code_detector = CodePlagiarismDetector(threshold=similarity_threshold_code, language=language_code_plag)
                plagiarism_manager.register_detector('code', code_detector)

                formatted_submissions_code = []
                for sub in code_submissions:
                    try:
                        # Filter by selected language
                        ext = sub['current_file']['extension']
                        file_lang = get_language_from_extension(ext)
                        if file_lang == language_code_plag:
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
                    lang_display = LANGUAGE_MAPPINGS[language_code_plag]['display']
                    st.success(f"Successfully compared {len(formatted_submissions_code)} {lang_display} submissions!")
                    st.rerun()
                else:
                    lang_display = LANGUAGE_MAPPINGS[language_code_plag]['display']
                    st.error(f"No {lang_display} code content could be processed for plagiarism detection.")

    if 'code' in st.session_state.plagiarism_results:
        display_code_plagiarism_results(st.session_state.plagiarism_results['code'])


def display_text_plagiarism_results(report):
    """Display text plagiarism results - showing ALL comparisons with highlighting for flagged ones."""
    st.markdown("#### Plagiarism Summary (Text)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Submissions", report['total_submissions'])
    with col2:
        st.metric("Total Comparisons", report.get('total_comparisons', 0))
    with col3:
        st.metric("Flagged Pairs", report['flagged_pairs'])
    with col4:
        st.metric("Threshold", f"{report['threshold'] * 100}%")

    if report['results']:
        st.markdown("#### All Similarity Results")
        st.markdown(f"🔴 **Red background** = Above threshold ({report['threshold'] * 100}%) - **Potential Plagiarism**")
        st.markdown("⚪ **White background** = Below threshold - **Acceptable Similarity**")

        # File summary section
        st.markdown("#### 📁 Files Being Compared")
        unique_files = set()
        for r in report['results']:
            if 'student1_file' in r:
                unique_files.add(f"{r['student1_name']}: {r['student1_file']}")
            if 'student2_file' in r:
                unique_files.add(f"{r['student2_name']}: {r['student2_file']}")

        if unique_files:
            st.write(f"**{len(unique_files)} unique student-file combinations:**")
            for file_info in sorted(unique_files):
                st.write(f"• {file_info}")

        st.markdown("---")

        # Create data for all results
        all_data = []
        for r in report['results']:
            all_data.append({
                'Student 1': r['student1_name'],
                'Student 1 ID': r['student1_id'],
                'File 1': r.get('student1_file', 'N/A'),
                'Student 2': r['student2_name'],
                'Student 2 ID': r['student2_id'],
                'File 2': r.get('student2_file', 'N/A'),
                'Overall Similarity': f"{r['similarity'] * 100:.2f}%",
                'TF-IDF Sim.': f"{r.get('tfidf_similarity', 0) * 100:.2f}%",
                'Fingerprint Sim.': f"{r.get('fingerprint_similarity', 0) * 100:.2f}%",
                'N-gram Sim.': f"{r.get('ngram_similarity', 0) * 100:.2f}%",
                'Status': '🚨 FLAGGED' if r['flagged'] else '✅ OK',
                'flagged': r['flagged']  # Hidden column for styling
            })

        if all_data:
            # Create DataFrame
            df = pd.DataFrame(all_data)

            # Apply styling and remove the hidden 'flagged' column for display
            display_df = df.drop('flagged', axis=1)

            # Style the dataframe
            styled_df = display_df.style.apply(
                lambda x: ['background-color: #ffcccc' if df.loc[x.name, 'flagged'] else '' for _ in x],
                axis=1
            ).applymap(
                lambda x: 'color: red; font-weight: bold' if '🚨 FLAGGED' in str(x) else
                ('color: green; font-weight: bold' if '✅ OK' in str(x) else ''),
                subset=['Status']
            )

            st.dataframe(styled_df, use_container_width=True)

            # Summary statistics
            if report['flagged_pairs'] > 0:
                st.markdown("#### Flagged Pairs Details")
                flagged_students = set()
                flagged_files = set()
                for r in report['results']:
                    if r['flagged']:
                        flagged_students.add(r['student1_name'])
                        flagged_students.add(r['student2_name'])
                        flagged_files.add(f"{r['student1_name']}: {r.get('student1_file', 'N/A')}")
                        flagged_files.add(f"{r['student2_name']}: {r.get('student2_file', 'N/A')}")

                st.error(f"⚠️ **{len(flagged_students)} students** involved in potential plagiarism:")
                st.write(", ".join(sorted(flagged_students)))

                st.markdown("**📄 Files involved in flagged comparisons:**")
                for file_info in sorted(flagged_files):
                    st.write(f"• {file_info}")
            else:
                st.success("🎉 No plagiarism detected above the threshold!")

            # Create similarity heatmap for all results
            create_similarity_heatmap(report['results'], "Text Plagiarism Similarity Matrix (All Comparisons)")
    else:
        st.info("No comparison results available.")


def display_code_plagiarism_results(report):
    """Display code plagiarism results - showing ALL comparisons with highlighting for flagged ones."""
    st.markdown("#### Plagiarism Summary (Code)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Submissions", report['total_submissions'])
    with col2:
        st.metric("Total Comparisons", report.get('total_comparisons', 0))
    with col3:
        st.metric("Flagged Pairs", report['flagged_pairs'])
    with col4:
        st.metric("Threshold", f"{report['threshold'] * 100}%")

    if report['results']:
        st.markdown("#### All Similarity Results")
        st.markdown(f"🔴 **Red background** = Above threshold ({report['threshold'] * 100}%) - **Potential Plagiarism**")
        st.markdown("⚪ **White background** = Below threshold - **Acceptable Similarity**")

        # File summary section
        st.markdown("#### 📁 Code Files Being Compared")
        unique_files = set()
        for r in report['results']:
            if 'student1_file' in r:
                unique_files.add(f"{r['student1_name']}: {r['student1_file']}")
            if 'student2_file' in r:
                unique_files.add(f"{r['student2_name']}: {r['student2_file']}")

        if unique_files:
            st.write(f"**{len(unique_files)} unique student-file combinations:**")
            for file_info in sorted(unique_files):
                st.write(f"• {file_info}")

        st.markdown("---")

        # Create data for all results
        all_data = []
        for r in report['results']:
            row = {
                'Student 1': r['student1_name'],
                'Student 1 ID': r['student1_id'],
                'File 1': r.get('student1_file', 'N/A'),
                'Student 2': r['student2_name'],
                'Student 2 ID': r['student2_id'],
                'File 2': r.get('student2_file', 'N/A'),
                'Overall Sim.': f"{r['similarity'] * 100:.2f}%",
                'Status': '🚨 FLAGGED' if r['flagged'] else '✅ OK',
                'flagged': r['flagged']  # Hidden column for styling
            }

            # Add available similarity metrics dynamically
            if 'token_similarity' in r:
                row['Token Sim.'] = f"{r['token_similarity'] * 100:.2f}%"
            if 'ast_similarity' in r:
                row['AST Sim.'] = f"{r['ast_similarity'] * 100:.2f}%"
            if 'structure_similarity' in r:
                row['Structure Sim.'] = f"{r['structure_similarity'] * 100:.2f}%"
            if 'normalized_similarity' in r:
                row['Normalized Sim.'] = f"{r['normalized_similarity'] * 100:.2f}%"
            if 'method_similarity' in r:
                row['Method Sim.'] = f"{r['method_similarity'] * 100:.2f}%"
            if 'class_similarity' in r:
                row['Class Sim.'] = f"{r['class_similarity'] * 100:.2f}%"
            if 'function_similarity' in r:
                row['Function Sim.'] = f"{r['function_similarity'] * 100:.2f}%"

            all_data.append(row)

        if all_data:
            # Create DataFrame
            df = pd.DataFrame(all_data)

            # Apply styling and remove the hidden 'flagged' column for display
            display_df = df.drop('flagged', axis=1)

            # Style the dataframe
            styled_df = display_df.style.apply(
                lambda x: ['background-color: #ffcccc' if df.loc[x.name, 'flagged'] else '' for _ in x],
                axis=1
            ).applymap(
                lambda x: 'color: red; font-weight: bold' if '🚨 FLAGGED' in str(x) else
                ('color: green; font-weight: bold' if '✅ OK' in str(x) else ''),
                subset=['Status']
            )

            st.dataframe(styled_df, use_container_width=True)

            # Summary statistics
            if report['flagged_pairs'] > 0:
                st.markdown("#### Flagged Pairs Details")
                flagged_students = set()
                flagged_files = set()
                for r in report['results']:
                    if r['flagged']:
                        flagged_students.add(r['student1_name'])
                        flagged_students.add(r['student2_name'])
                        flagged_files.add(f"{r['student1_name']}: {r.get('student1_file', 'N/A')}")
                        flagged_files.add(f"{r['student2_name']}: {r.get('student2_file', 'N/A')}")

                st.error(f"⚠️ **{len(flagged_students)} students** involved in potential plagiarism:")
                st.write(", ".join(sorted(flagged_students)))

                st.markdown("**📄 Code files involved in flagged comparisons:**")
                for file_info in sorted(flagged_files):
                    st.write(f"• {file_info}")
            else:
                st.success("🎉 No plagiarism detected above the threshold!")

            # Create similarity heatmap for all results
            create_similarity_heatmap(report['results'], "Code Plagiarism Similarity Matrix (All Comparisons)",
                                      color_scale='Reds')
    else:
        st.info("No comparison results available.")


def create_similarity_heatmap(results, title, color_scale='Blues'):
    """Create a similarity heatmap for plagiarism results with threshold visualization and file information."""
    if not results:
        return

    all_students = set()
    threshold = 0.8  # Default threshold
    student_file_map = {}  # Map student names to their files

    for r in results:
        student1_name = r['student1_name']
        student2_name = r['student2_name']
        all_students.add(student1_name)
        all_students.add(student2_name)

        # Store file information for each student
        student_file_map[student1_name] = r.get('student1_file', 'N/A')
        student_file_map[student2_name] = r.get('student2_file', 'N/A')

        # Try to get the actual threshold from results if available
        if 'threshold' in r:
            threshold = r['threshold']

    student_list = sorted(list(all_students))

    if len(student_list) > 1:
        heatmap_data = pd.DataFrame(0.0, index=student_list, columns=student_list)
        hover_data = pd.DataFrame('', index=student_list, columns=student_list, dtype=str)

        for r in results:
            similarity_val = r['similarity'] * 100
            student1 = r['student1_name']
            student2 = r['student2_name']
            file1 = r.get('student1_file', 'N/A')
            file2 = r.get('student2_file', 'N/A')

            heatmap_data.loc[student1, student2] = similarity_val
            heatmap_data.loc[student2, student1] = similarity_val

            # Create hover text with file information
            hover_text = f"{student1} ({file1}) vs {student2} ({file2})<br>Similarity: {similarity_val:.1f}%"
            hover_data.loc[student1, student2] = hover_text
            hover_data.loc[student2, student1] = hover_text

        # Set diagonal to 100 (self-similarity)
        for student in student_list:
            heatmap_data.loc[student, student] = 100
            file_name = student_file_map.get(student, 'N/A')
            hover_data.loc[student, student] = f"{student} ({file_name})<br>Self-comparison: 100%"

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Student", y="Student", color="Similarity %"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=color_scale,
            title=f"{title}<br><sub>Red areas (>{threshold * 100}%) indicate potential plagiarism</sub>",
            aspect="auto"
        )

        # Update layout for better readability
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title="Students",
            yaxis_title="Students"
        )

        # Add text annotations for high similarity values with file info
        for i, student1 in enumerate(student_list):
            for j, student2 in enumerate(student_list):
                if i != j:  # Skip diagonal
                    similarity_val = heatmap_data.loc[student1, student2]
                    if similarity_val >= threshold * 100:
                        fig.add_annotation(
                            x=j, y=i,
                            text=f"{similarity_val:.1f}%<br><sub>{student_file_map.get(student1, 'N/A')[:10]}... vs<br>{student_file_map.get(student2, 'N/A')[:10]}...</sub>",
                            showarrow=False,
                            font=dict(color="white", size=8, family="Arial Black")
                        )

        # Update hover template to show file information
        fig.update_traces(
            hovertemplate="<b>%{x} vs %{y}</b><br>" +
                          "Similarity: %{z:.1f}%<br>" +
                          "<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add a detailed comparison table for flagged pairs
        flagged_comparisons = [r for r in results if r['flagged']]
        if flagged_comparisons:
            st.markdown("#### 🚨 Detailed File Comparison for Flagged Pairs")
            detailed_data = []
            for r in flagged_comparisons:
                detailed_data.append({
                    'Student 1': r['student1_name'],
                    'File 1': r.get('student1_file', 'N/A'),
                    'Student 2': r['student2_name'],
                    'File 2': r.get('student2_file', 'N/A'),
                    'Similarity': f"{r['similarity'] * 100:.2f}%"
                })

            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(detailed_df, use_container_width=True)


def render_report_page():
    st.title("📊 Generate Reports")

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

    st.header("Configure Reports")

    with st.form("report_form"):
        st.markdown("#### Report Settings")
        assignment_name = st.text_input("Assignment Name", value="Assignment", help="Name for report filenames")

        st.markdown("#### Select Reports to Generate")
        generate_grade_csv = st.checkbox("Generate Grades CSV (for myITS Classroom import)",
                                         value=True, disabled=not st.session_state.grading_results)
        generate_plagiarism_html = st.checkbox("Generate Plagiarism Report (HTML)",
                                               value=True, disabled=not st.session_state.plagiarism_results)
        generate_summary = st.checkbox("Generate Overall Summary Report (JSON)", value=True)

        submitted_reports = st.form_submit_button("Generate Reports")

        if submitted_reports:
            if not (generate_grade_csv or generate_plagiarism_html or generate_summary):
                st.warning("Please select at least one type of report to generate.")
            else:
                with st.spinner("Generating reports..."):
                    report_generator = ReportGenerator()
                    reports_generated_paths = {}

                    valid_grading_results = [
                        res for res in st.session_state.grading_results
                        if isinstance(res, dict) and 'student_id' in res and 'grade' in res and
                           isinstance(res['grade'], dict) and 'points' in res['grade']
                    ]

                    if generate_grade_csv and valid_grading_results:
                        try:
                            grade_csv_path = report_generator.generate_grade_csv(assignment_name, valid_grading_results)
                            reports_generated_paths['grade_csv'] = grade_csv_path
                        except Exception as e:
                            st.error(f"Error generating grade CSV: {e}")

                    if generate_plagiarism_html and st.session_state.plagiarism_results:
                        for report_type, report_content in st.session_state.plagiarism_results.items():
                            if report_content and report_content.get('results'):
                                try:
                                    html_path = report_generator.generate_plagiarism_html(
                                        f"{assignment_name}_{report_type}_plagiarism", report_content
                                    )
                                    reports_generated_paths[f'plagiarism_html_{report_type}'] = html_path
                                except Exception as e:
                                    st.error(f"Error generating plagiarism HTML for {report_type}: {e}")

                    if generate_summary:
                        try:
                            summary_data = report_generator.generate_summary_report(
                                assignment_name, valid_grading_results, st.session_state.plagiarism_results
                            )
                            reports_generated_paths['summary_json'] = summary_data
                        except Exception as e:
                            st.error(f"Error generating summary report: {e}")

                    st.session_state.report_paths = reports_generated_paths
                    if reports_generated_paths:
                        st.success(f"Successfully generated {len(reports_generated_paths)} reports!")
                    st.rerun()

    # Display download links
    if st.session_state.report_paths:
        st.header("Download Reports")
        for report_key, report_item in st.session_state.report_paths.items():
            if report_key == 'summary_json':
                st.subheader("Summary Report Data")
                st.json(report_item)
            elif isinstance(report_item, str) and os.path.exists(report_item):
                report_name = os.path.basename(report_item)
                mime_type = 'text/csv' if report_name.endswith('.csv') else \
                    'text/html' if report_name.endswith('.html') else \
                        'application/json' if report_name.endswith('.json') else \
                            'application/octet-stream'
                try:
                    with open(report_item, 'rb') as f_report:
                        st.download_button(
                            label=f"📥 Download {report_name}",
                            data=f_report,
                            file_name=report_name,
                            mime=mime_type
                        )
                except Exception as e:
                    st.error(f"Could not prepare {report_name} for download: {e}")


# Main page rendering
current_page = st.session_state.current_page

if current_page == 'grade':
    render_grading_page()
elif current_page == 'plagiarism':
    render_plagiarism_page()
elif current_page == 'report':
    render_report_page()