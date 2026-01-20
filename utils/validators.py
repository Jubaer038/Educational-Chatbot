from __future__ import annotations

import datetime as dt
from typing import Optional

import streamlit as st


def validate_pdf(uploaded_file) -> Optional[bytes]:
    if uploaded_file is None:
        st.warning("Please upload a PDF file.")
        return None
    if not uploaded_file.name.lower().endswith(".pdf"):
        st.error("Only PDF files are supported.")
        return None
    data = uploaded_file.read()
    if not data:
        st.error("Uploaded PDF is empty.")
        return None
    return data


def validate_date(value: dt.date) -> bool:
    if value <= dt.date.today():
        st.error("Exam date must be in the future.")
        return False
    return True
