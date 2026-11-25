"""
prints the numbers quoted in the Results and Discussion that are not read off a figure

    python reported_numbers.py

The code is in d_plots/4_reported_numbers/reported_numbers.py; this file runs it from the top of the repository.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "d_plots" / "4_reported_numbers" / "reported_numbers.py"),
               run_name="__main__")
