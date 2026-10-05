"""Breathing stability from respiratory effort signals."""
from .api import BSIResult, Config, compute_bsi
from .io import compute_bsi_file

__version__ = "0.1.0"
__all__ = ["BSIResult", "Config", "compute_bsi", "compute_bsi_file"]
