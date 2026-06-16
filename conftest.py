"""
conftest.py - pytest configuration for the StyleFindr test suite.

Placing conftest.py at the project root puts the repo root on sys.path so the
tests can import top-level modules (tools, agent, app) directly when invoked
with `pytest tests/` from the project root. No fixtures are defined here;
file-level sys.path manipulation is the sole responsibility of this module.
"""
