import os
import time
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px

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
    st.session_state.current_page = 'upload'

if 'file_processor' not in st.session_state:
    st.session_state.file_processor = None

if 'submissions' not in st.session_state:
    st.session_state.submissions = None

if 'categorized_submissions' not in st.session_state:
    st.session_state.categorized_submissions = None

if 'grading_results' not in st.session_state:
    st.session_state.grading_results = []

if 'plagiarism_results' not in st.session_state:
    st.session_state.plagiarism_results = {}

if 'report_paths' not in st.session_state:
    st.session_state.report_paths = {}

# Sidebar navigation
st.sidebar.title("Navigation")

pages = {
    'upload': 'Upload Files',
    'grade': 'Auto-Grading',
    'plagiarism': 'Plagiarism Detection',
    'report': 'Generate Reports'
}

for page_id, page_name in pages.items():
    if st.sidebar.button(page_name):
        st.session_state.current_page = page_id

# Show current page name in sidebar
st.sidebar.markdown(f"**Current Page:** {pages[st.session_state.current_page]}")

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Auto-Grading & Plagiarism Detection Tool for myITS Classroom")
st.sidebar.markdown("v1.0.0")


# Upload Page
def render_upload_page():
    st.title("Upload Assignment Files")

    # File upload widget
    st.header("1. Upload myITS Classroom ZIP file")
    uploaded_file = st.file_uploader("Upload myITS Classroom assignment export ZIP", type=['zip'])

    if uploaded_file is not None:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            zip_path = tmp_file.name

        with st.spinner("Processing ZIP file..."):
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
            st.header("2. Submission Overview")
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
                for file in submission['files']:
                    ext = file['extension']
                    if ext in file_types:
                        file_types[ext] += 1
                    else:
                        file_types[ext] = 1

            # Create chart data
            chart_data = pd.DataFrame({
                'Extension': list(file_types.keys()),
                'Count': list(file_types.values())
            })

            # Display chart if we have data
            if not chart_data.empty:
                st.subheader("File Extensions")
                fig = px.bar(chart_data, x='Extension', y='Count', color='Extension')
                st.plotly_chart(fig)

            # Student table
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

            # Navigation
            st.markdown("---")
            st.markdown("**Next Steps:** Move to the Auto-Grading tab to set up grading criteria.")
            if st.button("Proceed to Auto-Grading"):
                st.session_state.current_page = 'grade'
                st.rerun()


# Auto-Grading Page
def render_grading_page():
    st.title("Auto-Grading")

    # Check if we have submissions
    if st.session_state.categorized_submissions is None:
        st.warning("No submissions loaded. Please upload files first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return

    # Assignment configuration
    st.header("Assignment Configuration")
    assignment_type = st.selectbox(
        "Assignment Type",
        options=["Essay", "Short Answer", "Code"],
        help="Select the type of assignment to apply appropriate grading criteria"
    )

    # Display available files to grade
    st.header("Grading Setup")

    if assignment_type == "Essay":
        st.subheader("Essay Grading")

        # Show how many text submissions we have
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions.")

        if len(text_submissions) > 0:
            # Text grading form
            with st.form("essay_grading_form"):
                st.markdown("#### Grading Criteria")

                # Answer key
                answer_key = st.text_area(
                    "Answer Key (Model Answer)",
                    height=200,
                    placeholder="Enter the model answer or rubric for grading essays..."
                )

                # Grading options
                col1, col2 = st.columns(2)
                with col1:
                    total_points = st.number_input("Total Points", min_value=1, value=100)
                with col2:
                    similarity_threshold = st.slider(
                        "Similarity Threshold for Full Points",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.7,
                        step=0.05,
                        help="Minimum similarity for full points (0.7 recommended)"
                    )

                # Submit button
                submitted = st.form_submit_button("Start Essay Grading")

                if submitted and answer_key:
                    with st.spinner("Grading essay submissions..."):
                        # Initialize grading manager
                        grading_manager = GradingManager()

                        # Register text grader
                        text_grader = TextGrader(
                            answer_key=answer_key,
                            total_points=total_points,
                            threshold=similarity_threshold
                        )
                        grading_manager.register_grader('text', text_grader)

                        # Grade each submission
                        results = []
                        for submission in text_submissions:
                            try:
                                # Get file content
                                file_path = submission['current_file']['path']
                                content = st.session_state.file_processor.get_file_content(file_path)

                                # Grade the submission
                                grade_result = grading_manager.grade_submission('text', content)

                                # Store the result
                                results.append({
                                    'student_id': submission['student_id'],
                                    'student_name': submission['student_name'],
                                    'file_name': submission['current_file']['name'],
                                    'grade': grade_result
                                })
                            except Exception as e:
                                st.error(f"Error grading {submission['student_name']}'s submission: {e}")

                        # Store in session state
                        st.session_state.grading_results = results

                        # Show success message
                        st.success(f"Successfully graded {len(results)} essay submissions!")

                        # Display results
                        if results:
                            result_data = []
                            for result in results:
                                result_data.append({
                                    'Student ID': result['student_id'],
                                    'Student Name': result['student_name'],
                                    'Points': result['grade']['points'],
                                    'Percentage': f"{result['grade']['percentage']}%",
                                    'Similarity': f"{result['grade']['similarity'] * 100:.2f}%"
                                })

                            # Create DataFrame and display
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df)

                            # Calculate statistics
                            avg_grade = sum(r['grade']['points'] for r in results) / len(results)
                            max_grade = max(r['grade']['points'] for r in results)
                            min_grade = min(r['grade']['points'] for r in results)

                            # Display statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Grade", f"{avg_grade:.2f}")
                            with col2:
                                st.metric("Highest Grade", f"{max_grade:.2f}")
                            with col3:
                                st.metric("Lowest Grade", f"{min_grade:.2f}")

                            # Create grade distribution chart
                            grades = [r['grade']['points'] for r in results]
                            fig = px.histogram(
                                x=grades,
                                nbins=10,
                                labels={'x': 'Points', 'y': 'Count'},
                                title='Grade Distribution'
                            )
                            st.plotly_chart(fig)

    elif assignment_type == "Short Answer":
        st.subheader("Short Answer Grading")

        # Show how many text submissions we have
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions.")

        if len(text_submissions) > 0:
            # Short answer grading form
            with st.form("short_answer_grading_form"):
                st.markdown("#### Grading Criteria")

                # Model answer
                answer_key = st.text_area(
                    "Model Answer",
                    height=100,
                    placeholder="Enter the model/reference answer"
                )

                # Keywords/concepts
                keywords = st.text_area(
                    "Key Concepts (one per line)",
                    height=100,
                    placeholder="Enter important keywords or concepts that should appear in the answer"
                )

                # Points allocation
                col1, col2 = st.columns(2)
                with col1:
                    total_points = st.number_input("Total Points", min_value=1, value=10)
                with col2:
                    keyword_weight = st.slider(
                        "Keyword Matching Weight (%)",
                        min_value=0,
                        max_value=100,
                        value=70,
                        help="How much of the grade should be based on keyword matching vs. overall similarity"
                    )

                # Submit button
                submitted = st.form_submit_button("Start Short Answer Grading")

                if submitted and answer_key and keywords:
                    with st.spinner("Grading short answer submissions..."):
                        # Parse keywords
                        keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]

                        # Initialize grading manager
                        grading_manager = GradingManager()

                        # Register short answer grader
                        short_answer_grader = ShortAnswerGrader(
                            answer_key=answer_key,
                            keywords=keyword_list,
                            total_points=total_points,
                            keyword_match_points=(keyword_weight / 100) * total_points,
                            similarity_points=((100 - keyword_weight) / 100) * total_points
                        )
                        grading_manager.register_grader('short_answer', short_answer_grader)

                        # Grade each submission
                        results = []
                        for submission in text_submissions:
                            try:
                                # Get file content
                                file_path = submission['current_file']['path']
                                content = st.session_state.file_processor.get_file_content(file_path)

                                # Grade the submission
                                grade_result = grading_manager.grade_submission('short_answer', content)

                                # Store the result
                                results.append({
                                    'student_id': submission['student_id'],
                                    'student_name': submission['student_name'],
                                    'file_name': submission['current_file']['name'],
                                    'grade': grade_result
                                })
                            except Exception as e:
                                st.error(f"Error grading {submission['student_name']}'s submission: {e}")

                        # Store in session state
                        st.session_state.grading_results = results

                        # Show success message
                        st.success(f"Successfully graded {len(results)} short answer submissions!")

                        # Display results
                        if results:
                            result_data = []
                            for result in results:
                                result_data.append({
                                    'Student ID': result['student_id'],
                                    'Student Name': result['student_name'],
                                    'Points': result['grade']['points'],
                                    'Percentage': f"{result['grade']['percentage']}%",
                                    'Keywords': f"{len(result['grade']['matched_keywords'])}/{len(result['grade']['matched_keywords']) + len(result['grade']['missing_keywords'])}"
                                })

                            # Create DataFrame and display
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df)

                            # Detailed view for a selected student
                            st.subheader("Detailed Grading")
                            selected_student = st.selectbox(
                                "Select student to view detailed feedback",
                                options=[r['student_name'] for r in results]
                            )

                            # Show detailed feedback for selected student
                            selected_result = next((r for r in results if r['student_name'] == selected_student), None)
                            if selected_result:
                                st.markdown(f"**Student:** {selected_result['student_name']}")
                                st.markdown(
                                    f"**Points:** {selected_result['grade']['points']}/{selected_result['grade']['max_points']}")

                                # Keyword matching
                                st.markdown("**Matched Keywords:**")
                                if selected_result['grade']['matched_keywords']:
                                    st.write(", ".join(selected_result['grade']['matched_keywords']))
                                else:
                                    st.write("None")

                                st.markdown("**Missing Keywords:**")
                                if selected_result['grade']['missing_keywords']:
                                    st.write(", ".join(selected_result['grade']['missing_keywords']))
                                else:
                                    st.write("None")

                                # Similarity
                                st.markdown(
                                    f"**Overall Similarity:** {selected_result['grade']['similarity'] * 100:.2f}%")

                                # Feedback
                                st.markdown("**Feedback:**")
                                st.write(selected_result['grade']['feedback'])

    elif assignment_type == "Code":
        st.subheader("Code Grading")

        # Show how many code submissions we have
        code_submissions = st.session_state.categorized_submissions['code']
        st.info(f"Found {len(code_submissions)} code submissions.")

        if len(code_submissions) > 0:
            st.warning("Code grading requires test cases. Please define test cases for automatic grading.")

            # Code grading form
            with st.form("code_grading_form"):
                st.markdown("#### Grading Criteria")

                # Reference implementation
                reference_code = st.text_area(
                    "Reference Implementation (Optional)",
                    height=200,
                    placeholder="Enter reference implementation (optional)..."
                )

                # Language selection
                language = st.selectbox(
                    "Programming Language",
                    options=["python", "java", "c", "cpp", "javascript"],
                    index=0
                )

                # Test cases
                st.markdown("#### Test Cases")
                st.markdown("Define test cases in JSON format. Example for Python:")
                st.code('''
[
    {
        "function_name": "add_numbers",
        "input": "2, 3",
        "expected_output": "5"
    },
    {
        "assertion": "self.assertEqual(submission.add_numbers(5, 5), 10)"
    }
]
                ''')

                test_cases_json = st.text_area(
                    "Test Cases (JSON)",
                    height=200,
                    placeholder="Enter test cases in JSON format..."
                )

                # Grading options
                total_points = st.number_input("Total Points", min_value=1, value=100)

                # Submit button
                submitted = st.form_submit_button("Start Code Grading")

                if submitted and test_cases_json:
                    with st.spinner("Grading code submissions..."):
                        try:
                            # Parse test cases
                            import json
                            test_cases = json.loads(test_cases_json)

                            # Initialize grading manager
                            grading_manager = GradingManager()

                            # Register code grader
                            code_grader = CodeGrader(
                                answer_key=reference_code,
                                test_cases=test_cases,
                                language=language,
                                total_points=total_points
                            )
                            grading_manager.register_grader('code', code_grader)

                            # Grade each submission
                            results = []
                            for submission in code_submissions:
                                try:
                                    # Get file content
                                    file_path = submission['current_file']['path']
                                    content = st.session_state.file_processor.get_file_content(file_path)

                                    # Grade the submission
                                    grade_result = grading_manager.grade_submission('code', content)

                                    # Store the result
                                    results.append({
                                        'student_id': submission['student_id'],
                                        'student_name': submission['student_name'],
                                        'file_name': submission['current_file']['name'],
                                        'grade': grade_result
                                    })
                                except Exception as e:
                                    st.error(f"Error grading {submission['student_name']}'s submission: {e}")

                            # Store in session state
                            st.session_state.grading_results = results

                            # Show success message
                            st.success(f"Successfully graded {len(results)} code submissions!")

                            # Display results
                            if results:
                                result_data = []
                                for result in results:
                                    result_data.append({
                                        'Student ID': result['student_id'],
                                        'Student Name': result['student_name'],
                                        'Points': result['grade']['points'],
                                        'Percentage': f"{result['grade']['percentage']}%",
                                        'Passed Tests': f"{result['grade']['passed_tests']}/{result['grade']['total_tests']}"
                                    })

                                # Create DataFrame and display
                                result_df = pd.DataFrame(result_data)
                                st.dataframe(result_df)

                                # Show test details for a student
                                st.subheader("Test Details")
                                selected_student = st.selectbox(
                                    "Select student to view test results",
                                    options=[r['student_name'] for r in results]
                                )

                                # Show test results for selected student
                                selected_result = next((r for r in results if r['student_name'] == selected_student),
                                                       None)
                                if selected_result:
                                    st.markdown(f"**Student:** {selected_result['student_name']}")
                                    st.markdown(
                                        f"**Points:** {selected_result['grade']['points']}/{selected_result['grade']['max_points']}")

                                    # Test results
                                    st.markdown("**Test Results:**")
                                    for test in selected_result['grade']['test_results']:
                                        if test['passed']:
                                            st.success(f"✅ {test['name']} - Passed")
                                        else:
                                            st.error(f"❌ {test['name']} - Failed: {test['message']}")

                                # Calculate statistics
                                avg_grade = sum(r['grade']['points'] for r in results) / len(results)
                                max_grade = max(r['grade']['points'] for r in results)
                                min_grade = min(r['grade']['points'] for r in results)

                                # Display statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Grade", f"{avg_grade:.2f}")
                                with col2:
                                    st.metric("Highest Grade", f"{max_grade:.2f}")
                                with col3:
                                    st.metric("Lowest Grade", f"{min_grade:.2f}")

                                # Create grade distribution chart
                                grades = [r['grade']['points'] for r in results]
                                fig = px.histogram(
                                    x=grades,
                                    nbins=10,
                                    labels={'x': 'Points', 'y': 'Count'},
                                    title='Grade Distribution'
                                )
                                st.plotly_chart(fig)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format for test cases. Please check your input.")

    # Navigation
    st.markdown("---")
    st.markdown("**Next Steps:** Move to the Plagiarism Detection tab to check for similarities.")
    if st.button("Proceed to Plagiarism Detection"):
        st.session_state.current_page = 'plagiarism'
        st.rerun()


# Plagiarism Detection Page
def render_plagiarism_page():
    st.title("Plagiarism Detection")

    # Check if we have submissions
    if st.session_state.categorized_submissions is None:
        st.warning("No submissions loaded. Please upload files first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return

    # Display available files to check for plagiarism
    st.header("1. Select Submission Type for Plagiarism Detection")

    tab1, tab2 = st.tabs(["Text/Essay", "Code"])

    with tab1:
        st.subheader("Text Plagiarism Detection")

        # Show how many text submissions we have
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions.")

        if len(text_submissions) > 0:
            # Text plagiarism form
            with st.form("text_plagiarism_form"):
                st.markdown("#### Detection Settings")

                # Detection options
                similarity_threshold = st.slider(
                    "Similarity Threshold to Flag Plagiarism",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help="Minimum similarity to flag as potential plagiarism (0.8 recommended)"
                )

                # Submit button
                submitted = st.form_submit_button("Start Text Plagiarism Detection")

                if submitted:
                    with st.spinner("Detecting plagiarism in text submissions..."):
                        # Initialize plagiarism manager
                        plagiarism_manager = PlagiarismManager()

                        # Register text detector
                        text_detector = TextPlagiarismDetector(threshold=similarity_threshold)
                        plagiarism_manager.register_detector('text', text_detector)

                        # Detect plagiarism
                        report = plagiarism_manager.detect_plagiarism('text', text_submissions)

                        # Store in session state
                        st.session_state.plagiarism_results['text'] = report

                        # Show success message
                        st.success(f"Successfully compared {len(text_submissions)} text submissions!")

                        # Display results
                        if report:
                            st.markdown(f"#### Plagiarism Summary")
                            st.markdown(f"Total Submissions: {report['total_submissions']}")
                            st.markdown(f"Flagged Pairs: {report['flagged_pairs']}")
                            st.markdown(f"Flagged Students: {report['flagged_students']}")
                            st.markdown(f"Similarity Threshold: {report['threshold'] * 100}%")

                            # Show flagged pairs
                            if report['flagged_pairs'] > 0:
                                st.markdown("#### Flagged Submissions")

                                flagged_data = []
                                for result in report['results']:
                                    if result['flagged']:
                                        flagged_data.append({
                                            'Student 1': f"{result['student1_name']} ({result['student1_id']})",
                                            'Student 2': f"{result['student2_name']} ({result['student2_id']})",
                                            'Similarity': f"{result['similarity'] * 100:.2f}%",
                                            'File 1': result['student1_file'],
                                            'File 2': result['student2_file']
                                        })

                                # Create DataFrame and display
                                if flagged_data:
                                    flagged_df = pd.DataFrame(flagged_data)
                                    st.dataframe(flagged_df)

                                    # Create heatmap data
                                    students = set()
                                    for result in report['results']:
                                        students.add(result['student1_name'])
                                        students.add(result['student2_name'])

                                    students = list(students)
                                    heatmap_data = pd.DataFrame(0, index=students, columns=students)

                                    for result in report['results']:
                                        s1 = result['student1_name']
                                        s2 = result['student2_name']
                                        heatmap_data.loc[s1, s2] = result['similarity'] * 100
                                        heatmap_data.loc[s2, s1] = result['similarity'] * 100

                                    # Create heatmap
                                    fig = px.imshow(
                                        heatmap_data,
                                        labels=dict(x="Student", y="Student", color="Similarity %"),
                                        x=heatmap_data.columns,
                                        y=heatmap_data.index,
                                        color_continuous_scale='Blues'
                                    )
                                    st.plotly_chart(fig)
                            else:
                                st.success("No plagiarism detected above the threshold!")

    with tab2:
        st.subheader("Code Plagiarism Detection")

        # Show how many code submissions we have
        code_submissions = st.session_state.categorized_submissions['code']
        st.info(f"Found {len(code_submissions)} code submissions.")

        if len(code_submissions) > 0:
            # Code plagiarism form
            with st.form("code_plagiarism_form"):
                st.markdown("#### Detection Settings")

                # Language selection
                language = st.selectbox(
                    "Programming Language",
                    options=["python", "java", "c", "cpp", "javascript"],
                    index=0
                )

                # Detection options
                similarity_threshold = st.slider(
                    "Similarity Threshold to Flag Plagiarism",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help="Minimum similarity to flag as potential plagiarism (0.8 recommended)"
                )

                # Submit button
                submitted = st.form_submit_button("Start Code Plagiarism Detection")

                if submitted:
                    with st.spinner("Detecting plagiarism in code submissions..."):
                        # Initialize plagiarism manager
                        plagiarism_manager = PlagiarismManager()

                        # Register code detector
                        code_detector = CodePlagiarismDetector(
                            threshold=similarity_threshold,
                            language=language
                        )
                        plagiarism_manager.register_detector('code', code_detector)

                        # Detect plagiarism
                        report = plagiarism_manager.detect_plagiarism('code', code_submissions)

                        # Store in session state
                        st.session_state.plagiarism_results['code'] = report

                        # Show success message
                        st.success(f"Successfully compared {len(code_submissions)} code submissions!")

                        # Display results
                        if report:
                            st.markdown(f"#### Plagiarism Summary")
                            st.markdown(f"Total Submissions: {report['total_submissions']}")
                            st.markdown(f"Flagged Pairs: {report['flagged_pairs']}")
                            st.markdown(f"Flagged Students: {report['flagged_students']}")
                            st.markdown(f"Similarity Threshold: {report['threshold'] * 100}%")

                            # Show flagged pairs
                            if report['flagged_pairs'] > 0:
                                st.markdown("#### Flagged Submissions")

                                flagged_data = []
                                for result in report['results']:
                                    if result['flagged']:
                                        flagged_data.append({
                                            'Student 1': f"{result['student1_name']} ({result['student1_id']})",
                                            'Student 2': f"{result['student2_name']} ({result['student2_id']})",
                                            'Similarity': f"{result['similarity'] * 100:.2f}%",
                                            'Token Sim': f"{result['token_similarity'] * 100:.2f}%",
                                            'AST Sim': f"{result['ast_similarity'] * 100:.2f}%"
                                        })

                                # Create DataFrame and display
                                if flagged_data:
                                    flagged_df = pd.DataFrame(flagged_data)
                                    st.dataframe(flagged_df)

                                    # Create heatmap data
                                    students = set()
                                    for result in report['results']:
                                        students.add(result['student1_name'])
                                        students.add(result['student2_name'])

                                    students = list(students)
                                    heatmap_data = pd.DataFrame(0, index=students, columns=students)

                                    for result in report['results']:
                                        s1 = result['student1_name']
                                        s2 = result['student2_name']
                                        heatmap_data.loc[s1, s2] = result['similarity'] * 100
                                        heatmap_data.loc[s2, s1] = result['similarity'] * 100

                                    # Create heatmap
                                    fig = px.imshow(
                                        heatmap_data,
                                        labels=dict(x="Student", y="Student", color="Similarity %"),
                                        x=heatmap_data.columns,
                                        y=heatmap_data.index,
                                        color_continuous_scale='Reds'
                                    )
                                    st.plotly_chart(fig)
                            else:
                                st.success("No plagiarism detected above the threshold!")

    # Navigation
    st.markdown("---")
    st.markdown("**Next Steps:** Move to the Generate Reports tab to create CSV grade reports and plagiarism reports.")
    if st.button("Proceed to Generate Reports"):
        st.session_state.current_page = 'report'
        st.rerun()


# Report Generation Page
def render_report_page():
    st.title("Generate Reports")

    # Check if we have grading results
    if not st.session_state.grading_results:
        st.warning("No grading results available. Please grade submissions first.")
        if st.button("Go to Auto-Grading Page"):
            st.session_state.current_page = 'grade'
            st.rerun()
        return

    st.header("1. Configure Reports")

    # Report generation form
    with st.form("report_form"):
        st.markdown("#### Report Settings")

        # Assignment name
        assignment_name = st.text_input(
            "Assignment Name",
            value="Assignment",
            help="Name of the assignment for the report filenames"
        )

        # Report types
        st.markdown("#### Select Reports to Generate")

        generate_grade_csv = st.checkbox("Generate Grades CSV (for myITS Classroom import)", value=True)
        generate_plagiarism_html = st.checkbox("Generate Plagiarism Report (HTML)", value=True)
        generate_summary = st.checkbox("Generate Summary Report (JSON)", value=True)

        # Submit button
        submitted = st.form_submit_button("Generate Reports")

        if submitted:
            with st.spinner("Generating reports..."):
                # Initialize report generator
                report_generator = ReportGenerator()

                # Generate reports
                reports = {}

                if generate_grade_csv:
                    # Generate grade CSV
                    grade_csv_path = report_generator.generate_grade_csv(
                        assignment_name,
                        st.session_state.grading_results
                    )
                    reports['grade_csv'] = grade_csv_path

                if generate_plagiarism_html:
                    # Generate plagiarism HTML for each type
                    for report_type, report in st.session_state.plagiarism_results.items():
                        html_path = report_generator.generate_plagiarism_html(
                            f"{assignment_name}_{report_type}",
                            report
                        )
                        reports[f'plagiarism_html_{report_type}'] = html_path

                if generate_summary:
                    # Generate summary report
                    plagiarism_report = None
                    if st.session_state.plagiarism_results:
                        # Use the first report as the summary
                        plagiarism_report = next(iter(st.session_state.plagiarism_results.values()))

                    summary = report_generator.generate_summary_report(
                        assignment_name,
                        st.session_state.grading_results,
                        plagiarism_report
                    )
                    reports['summary'] = summary

                # Store in session state
                st.session_state.report_paths = reports

                # Show success message
                st.success(f"Successfully generated {len(reports)} reports!")

    # Display reports if available
    if st.session_state.report_paths:
        st.header("2. Download Reports")

        for report_type, report_path in st.session_state.report_paths.items():
            if report_type == 'summary':
                # For summary, just show the data
                st.subheader("Summary Report")
                st.json(report_path)
            else:
                # For files, create download links
                report_name = os.path.basename(report_path)

                with open(report_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {report_name}",
                        data=f,
                        file_name=report_name,
                        mime='application/octet-stream'
                    )


# Render the current page
current_page = st.session_state.current_page

if current_page == 'upload':
    render_upload_page()
elif current_page == 'grade':
    render_grading_page()
elif current_page == 'plagiarism':
    render_plagiarism_page()
elif current_page == 'report':
    render_report_page()