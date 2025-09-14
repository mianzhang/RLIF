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

"""IFTrain instruction following evaluation package for VERL reward scoring."""

import json
import os
from typing import Dict, List, Any, Union

from .evaluation import evaluate_instruction_following, compute_score_internal
from .utils import import_iftrain_modules


def compute_score(solution_str: str, ground_truth: Union[str, Dict], strict: bool = True) -> Dict[str, Any]:
    """
    Compute the IFTrain instruction following score.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: Either a JSON string or dict containing instruction information
        strict: Whether to use strict evaluation mode
        
    Returns:
        Dictionary containing the score and evaluation details
    """
    return compute_score_internal(solution_str, ground_truth, strict)


def compute_score_loose(solution_str: str, ground_truth: Union[str, Dict]) -> Dict[str, Any]:
    """
    Compute the IFTrain instruction following score using loose evaluation.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: Either a JSON string or dict containing instruction information
        
    Returns:
        Dictionary containing the score and evaluation details
    """
    return compute_score(solution_str, ground_truth, strict=False)


# Export main functions
__all__ = ['compute_score', 'compute_score_loose', 'evaluate_instruction_following'] 