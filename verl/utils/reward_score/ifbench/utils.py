# Copyright 2024 Bytedance Ltd. and/or its affiliates

"""Utility functions for IFBench evaluation."""

import importlib.util
import os


def import_ifbench_modules():
    """Import IFBench evaluation modules from the package."""
    try:
        # Import modules directly from the package
        from . import instructions_registry
        from . import evaluation_lib

        return instructions_registry, evaluation_lib

    except ImportError as e:
        raise ImportError(f"Failed to import IFBench modules: {e}")


class MockInputExample:
    """Mock InputExample class for IFBench evaluation."""
    
    def __init__(self, instruction_id_list, kwargs, prompt=""):
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs
        self.prompt = prompt 