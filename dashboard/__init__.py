"""Streamlit front-end for HGM runs (entry point: ``hgm_dashboard.py``).

Presentation only -- every byte of data comes from ``meta_agent.run_inspect``
and ``meta_agent.run_inspect_agentic`` (pure Python, unit-tested). Split into
``loaders`` (``st.cache_data`` wrappers keyed on file mtimes), ``components``
(widgets shared by several views) and one module per view.
"""
