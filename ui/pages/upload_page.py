import os
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px

from core.file_processor import FileProcessor
from utils.utils import navigate_to_page

def render_upload_page():
    st.title("Upload Assignment Files")

    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False

    # File upload widget
    st.header("1. Upload myITS Classroom ZIP file")
    uploaded_file = st.file_uploader("Upload myITS Classroom assignment export ZIP", type=['zip'])

    if uploaded_file is not None and not st.session_state.processing_done:
        if st.button("Submit and Process Files", type="primary"):
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
                    st.session_state.processing_done = True
                    
                    # Force a rerun to show results
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing ZIP file: {e}")
                finally:
                    # Clean up temporary file
                    if 'zip_path' in locals():
                        try:
                            os.unlink(zip_path)
                        except Exception:
                            pass

    # Show results only after processing is done
    if st.session_state.processing_done:
        # Show success message
        st.success(f"Successfully processed ZIP file! Found {len(st.session_state.submissions)} student submissions.")

        # Display submission stats
        st.header("2. Submission Overview")
        st.markdown(f"**Total Submissions:** {len(st.session_state.submissions)}")

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
        for submission in st.session_state.submissions:
            for file in submission['files']:
                ext = file['extension']
                file_types[ext] = file_types.get(ext, 0) + 1

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

        student_data = [
            {
                'Student ID': submission['student_id'],
                'Student Name': submission['student_name'],
                'Files': len(submission['files'])
            }
            for submission in st.session_state.submissions
        ]

        student_df = pd.DataFrame(student_data)
        st.dataframe(student_df, use_container_width=True, hide_index=True)

        # Navigation
        st.markdown("---")
        st.markdown("**Next Steps:** Move to the Auto-Grading tab to set up grading criteria.")
        if st.button("Proceed to Auto-Grading"):
            st.session_state.processing_done = False
            navigate_to_page('grade')