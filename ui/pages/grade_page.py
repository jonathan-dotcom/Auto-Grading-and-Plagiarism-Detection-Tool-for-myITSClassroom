import json
import streamlit as st
import pandas as pd
import plotly.express as px

from core.grader import TextGrader, CodeGrader, GradingManager
from utils.utils import check_submissions_loaded, navigate_to_page

def render_grading_page():
    st.title("Auto-Grading")

    # Check if submissions are loaded
    if not check_submissions_loaded():
        return

    # Display available files to grade
    st.header("1. Select Assignment Type to Grade")

    tab1, tab2 = st.tabs(["Text/Essay", "Code"])

    with tab1:
        st.subheader("Text/Essay Grading")

        # Show how many text submissions we have
        text_submissions = st.session_state.categorized_submissions['text']
        st.info(f"Found {len(text_submissions)} text submissions.")

        if len(text_submissions) > 0:
            # Text grading form
            with st.form("text_grading_form"):
                st.markdown("#### Grading Criteria")

                # Answer key
                answer_key = st.text_area(
                    "Answer Key (Model Answer)",
                    height=200,
                    placeholder="Enter the model answer or rubric for grading text submissions..."
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
                submitted = st.form_submit_button("Start Text Grading")

                if submitted and answer_key:
                    with st.spinner("Grading text submissions..."):
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
                        st.success(f"Successfully graded {len(results)} text submissions!")

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

    with tab2:
        st.subheader("Code Grading")

        # Similar structure to text grading, but with code-specific logic
        # You would implement this similarly to the text grading section
        # with appropriate modifications for code submissions

    # Navigation
    st.markdown("---")
    st.markdown("**Next Steps:** Move to the Plagiarism Detection tab to check for similarities.")
    if st.button("Proceed to Plagiarism Detection"):
        navigate_to_page('plagiarism')