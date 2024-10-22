import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
from pathlib import Path

st.set_page_config(layout="wide")

# Get the current file's directory
current_dir = Path(__file__).resolve().parent

# Construct the path to the pages.toml file
pages_toml_path = current_dir / '.streamlit' / 'pages.toml'

nav = get_nav_from_toml(str(pages_toml_path))

pg = st.navigation(nav)

pg.run()
