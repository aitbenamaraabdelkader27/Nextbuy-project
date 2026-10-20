"""Streamlit Community Cloud entry point. Run from any working directory."""

import streamlit as st

from nextbuy_core.ui import main

st.set_page_config(page_title="NextBuy · Grocery Intelligence", page_icon="◈", layout="wide")
main()
