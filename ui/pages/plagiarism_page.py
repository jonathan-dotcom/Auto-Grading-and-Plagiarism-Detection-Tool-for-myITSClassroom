import streamlit as st
import pandas as pd
import plotly.express as px

from core.plagiarism import TextPlagiarismDetector, CodePlagiarismDetector, PlagiarismManager
from utils.utils import check_submissions_loaded, navigate_to_page

def render_plagiarism_page():
    st.title("Plagiarism Detection")

    # Check if submissions are loaded
    if not check_submissions_loaded():
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
        # Similar structure to text plagiarism detection
        # Implement code plagiarism detection logic here

    # Navigation
    st.markdown("---")
    st.markdown("**Next Steps:** Move to the Generate Reports tab to create CSV grade reports and plagiarism reports.")
    if st.button("Proceed to Generate Reports"):
        navigate_to_page('report')