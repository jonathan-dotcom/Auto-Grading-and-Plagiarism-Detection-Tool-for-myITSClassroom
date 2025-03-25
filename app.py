import streamlit as st

# Import page rendering functions
from ui.pages.upload_page import render_upload_page
from ui.pages.grade_page import render_grading_page
from ui.pages.plagiarism_page import render_plagiarism_page
from ui.pages.report_page import render_report_page
from utils.utils import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="MyITS Auto-Grader & Plagiarism Detector",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar():
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

    st.sidebar.markdown(f"**Current Page:** {pages[st.session_state.current_page]}")

    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("Auto-Grading & Plagiarism Detection Tool for myITS Classroom")
    st.sidebar.markdown("v1.0.0")

def main():
    # Initialize session state
    initialize_session_state()

    # Create sidebar
    create_sidebar()

    # Render current page
    current_page = st.session_state.current_page
    
    if current_page == 'upload':
        render_upload_page()
    elif current_page == 'grade':
        render_grading_page()
    elif current_page == 'plagiarism':
        render_plagiarism_page()
    elif current_page == 'report':
        render_report_page()

if __name__ == "__main__":
    main()