"""Compatibility launcher. Prefer `streamlit run app.py` from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextbuy_core.ui import main
import streamlit as st

st.set_page_config(page_title="NextBuy · Grocery Intelligence", page_icon="◈", layout="wide")
main()
