"""
Parses raw SAWS .xls files into a unified long-format DataFrame.

Each SAWS file contains one sheet with a legend block at the top followed by
one block per station per year. Each block is a Day×Month table for a single
meteorological variable. Multiple variables (e.g. max temp, min temp, rain)
are stacked sequentially within the same file.

Output: a long-format DataFrame indexed by (station_id, date, variable),
ready to be passed to impute.py.
"""

import pandas as pd


def parse_saws_file(filepath: str) -> pd.DataFrame:
    """Read a single SAWS .xls file and return a long-format DataFrame."""
    raise NotImplementedError


def parse_saws_directory(directory: str) -> pd.DataFrame:
    """Parse all .xls files in directory and concatenate into one DataFrame."""
    raise NotImplementedError
