"""Utility functions for IFTrain."""

import importlib.util
import os


def import_iftrain_modules():
    """Import IFTrain evaluation modules from the package."""
    try:
        # Import modules directly from the package
        from . import instructions_registry
        from . import evaluation_lib

        return instructions_registry, evaluation_lib

    except ImportError as e:
        raise ImportError(f"Failed to import IFTrain modules: {e}")


class MockInputExample:
    """Mock InputExample class for IFTrain evaluation."""
    
    def __init__(self, instruction_id_list, kwargs, prompt=""):
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs
        self.prompt = prompt 