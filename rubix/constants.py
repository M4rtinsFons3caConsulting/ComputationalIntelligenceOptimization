"""
constants.py - this module centralizes the project's constants.

PROJECT TREE - which are the constant values for the project tree, to facilitate imports and all things 
path related.

"""

from pathlib import Path

# ----------- PROJECT TREE -------------- #
# Stores path logic for the project, anchoring at ROOT.

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data" 
DATA_V1 =  DATA_DIR / "player_data.xlsx"
DATA_V2 =  DATA_DIR / "player_data(Copy).xlsx"
