from __future__ import annotations

from typing import Callable, Any

import streamlit as st


def safe_call(fn: Callable[[], Any], fallback_message: str = "Something went wrong.") -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover
        st.error(f"{fallback_message} Details: {exc}")
        return None
