# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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