"""Test configuration.

The suite tests the built-in formant engine unless a test asks for another engine by name, so
its results do not depend on which neural models happen to be installed on the machine.
"""
import os

os.environ["SYNTH2_ENGINE"] = "formant"
