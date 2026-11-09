"""Test configuration.

The suite tests the built-in formant engine unless a test asks for another engine by name, so
its results do not depend on which neural models happen to be installed on the machine.
"""
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:  # lets "python -I -m pytest" find the package without installing it
    sys.path.insert(0, ROOT)

os.environ["SYNTH2_ENGINE"] = "formant"
