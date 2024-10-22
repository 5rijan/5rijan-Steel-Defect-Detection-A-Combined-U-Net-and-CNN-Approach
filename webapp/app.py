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

# Add the title to the current page
add_page_title()

# Main content area
st.write("Welcome to the main page. Please select a page from the sidebar.")
