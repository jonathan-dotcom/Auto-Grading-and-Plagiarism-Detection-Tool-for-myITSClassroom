import os
import streamlit as st
import pandas as pd

from core.report_generator import ReportGenerator
from utils.utils import check_submissions_loaded

def render_report_page():
    st.title("Generate Reports")

    # Check if we have grading results
    if not st.session_state.grading_results:
        st.warning("No grading results available. Please grade submissions first.")
        if st.button("Go to Auto-Grading Page"):
            st.session_state.current_page = 'grade'
            st.experimental_rerun()
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