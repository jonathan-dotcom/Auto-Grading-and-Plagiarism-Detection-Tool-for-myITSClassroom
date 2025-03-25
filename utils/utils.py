import streamlit as st

def initialize_session_state():
    """Initialize session state with default values if not already set."""
    defaults = {
        'current_page': 'upload',
        'file_processor': None,
        'submissions': None,
        'categorized_submissions': None,
        'grading_results': [],
        'plagiarism_results': {},
        'report_paths': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_submissions_loaded():
    """
    Check if submissions are loaded.
    Returns True if submissions are available, False otherwise.
    Provides a warning and option to return to upload page if not.
    """
    if st.session_state.categorized_submissions is None:
        st.warning("No submissions loaded. Please upload files first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.experimental_rerun()
        return False
    return True

def navigate_to_page(page):
    """
    Helper function to navigate between pages.
    
    Args:
        page (str): Target page to navigate to.
    """
    st.session_state.current_page = page
    st.experimental_rerun()