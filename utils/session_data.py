import streamlit as st

from utils.load_data import load_data


def load_user_data(key: str, filename: str):
    if key not in st.session_state:
        st.session_state[key] = load_data(filename)
    return st.session_state[key]
