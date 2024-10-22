import streamlit as st
from st_pages import Page, show_pages, add_page_title

st.set_page_config(layout="wide")

# Define the pages
pages = [
    Page("pages/introduction.py", "Introduction", "📖"),
    Page("webapp/cnn.py", "CNN", "📊"),
    Page("webapp/unet.py", "U-Net", "📄"),
]

# Show the pages in the sidebar
show_pages(pages)

# Redirect to the introduction page
import streamlit.components.v1 as components

components.html(
    """
    <script>
    if (window.location.pathname === '/') {
        window.location.href = '?page=introduction';
    }
    </script>
    """,
    height=0,
)

# The rest of your app.py content (if any) goes here
