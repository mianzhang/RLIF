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

"""IFEval instruction following evaluation for VERL reward scoring."""

import json
import os
import sys
from typing import Dict, List, Any, Union

# Add the IFEval directory to Python path for imports
_IFEVAL_PATH = None

def _setup_ifeval_path():
    """Setup the path to IFEval evaluation code."""
    global _IFEVAL_PATH
    if _IFEVAL_PATH is None:
        # Try to find IFEval in the RLIF_data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to find RLIF_data/IFEval
        for _ in range(10):  # Limit search depth
            rlif_data_path = os.path.join(current_dir, 'RLIF_data', 'IFEval')
            if os.path.exists(rlif_data_path):
                _IFEVAL_PATH = rlif_data_path
                break
            current_dir = os.path.dirname(current_dir)
        
        if _IFEVAL_PATH is None:
            raise ImportError("Could not find RLIF_data/IFEval directory. Please ensure it exists.")
        
        # Add to path but ensure it doesn't override standard library modules
        if _IFEVAL_PATH not in sys.path:
            # Insert near the end to avoid conflicts with standard library
            sys.path.insert(-1, _IFEVAL_PATH)

def _import_ifeval_modules():
    """Import IFEval evaluation modules."""
    _setup_ifeval_path()
    
    # Temporarily modify sys.path to avoid conflicts
    original_path = sys.path.copy()
    try:
        # Remove the current reward_score directory from path temporarily
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir in sys.path:
            sys.path.remove(current_dir)
        
        # Add the parent directory to make IFEval a package
        global _IFEVAL_PATH
        if _IFEVAL_PATH:
            parent_dir = os.path.dirname(_IFEVAL_PATH)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
        
        # Now try importing with the IFEval prefix
        from IFEval import instructions_registry
        from IFEval import evaluation_lib
        return instructions_registry, evaluation_lib
    except ImportError as e:
        # Fallback: try importing without prefix
        try:
            import instructions_registry
            import evaluation_lib
            return instructions_registry, evaluation_lib
        except ImportError:
            raise ImportError(f"Failed to import IFEval modules: {e}")
    finally:
        # Restore original path
        sys.path[:] = original_path

def evaluate_instruction_following(response: str, instruction_ids: List[str], kwargs_list: List[Dict], 
                                 prompt: str = "", strict: bool = True) -> Dict[str, Any]:
    """
    Evaluate whether a response follows the given instructions.
    
    Args:
        response: The response text to evaluate
        instruction_ids: List of instruction IDs to check
        kwargs_list: List of kwargs dictionaries for each instruction
        prompt: The original prompt (used by some instructions)
        strict: Whether to use strict evaluation (True) or loose evaluation (False)
        
    Returns:
        Dictionary containing evaluation results
    """
    instructions_registry, evaluation_lib = _import_ifeval_modules()
    
    # Create an InputExample-like object
    class MockInputExample:
        def __init__(self, instruction_id_list, kwargs, prompt=""):
            self.instruction_id_list = instruction_id_list
            self.kwargs = kwargs
            self.prompt = prompt
    
    # Create mock input
    inp = MockInputExample(instruction_ids, kwargs_list, prompt)
    
    # Create prompt to response mapping
    prompt_to_response = {prompt: response}
    
    # Evaluate using the appropriate method
    if strict:
        result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    
    return {
        'instruction_id_list': result.instruction_id_list,
        'prompt': result.prompt,
        'response': result.response,
        'follow_all_instructions': result.follow_all_instructions,
        'follow_instruction_list': result.follow_instruction_list,
        'num_instructions': len(instruction_ids),
        'num_followed': sum(result.follow_instruction_list),
        'accuracy': sum(result.follow_instruction_list) / len(instruction_ids) if instruction_ids else 0.0
    }

def compute_score(solution_str: str, ground_truth: Union[str, Dict], strict: bool = True) -> Dict[str, Any]:
    """
    Compute the IFEval instruction following score.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: Either a JSON string or dict containing instruction information
        strict: Whether to use strict evaluation mode
        
    Returns:
        Dictionary containing the score and evaluation details
    """
    try:
        # Parse ground truth if it's a string
        if isinstance(ground_truth, str):
            gt_data = json.loads(ground_truth)
        else:
            gt_data = ground_truth
            
        # Extract instruction information
        instruction_ids = gt_data.get('instruction_ids', [])
        kwargs_list = gt_data.get('kwargs', [])
        prompt = gt_data.get('prompt', '')
        
        if not instruction_ids:
            # No instructions to check, return perfect score
            return {
                'score': 1.0,
                'acc': True,
                'follow_all_instructions': True,
                'follow_instruction_list': [],
                'num_instructions': 0,
                'num_followed': 0,
                'accuracy': 1.0,
                'evaluation_mode': 'strict' if strict else 'loose'
            }
        
        # Evaluate instruction following
        eval_result = evaluate_instruction_following(
            response=solution_str,
            instruction_ids=instruction_ids,
            kwargs_list=kwargs_list,
            prompt=prompt,
            strict=strict
        )
        
        # Calculate score
        # Use binary score: 1.0 if all instructions followed, 0.0 otherwise
        score = 1.0 if eval_result['follow_all_instructions'] else 0.0
        
        return {
            'score': score,
            'acc': eval_result['follow_all_instructions'],
            'follow_all_instructions': eval_result['follow_all_instructions'],
            'follow_instruction_list': eval_result['follow_instruction_list'],
            'num_instructions': eval_result['num_instructions'],
            'num_followed': eval_result['num_followed'],
            'accuracy': eval_result['accuracy'],
            'evaluation_mode': 'strict' if strict else 'loose',
            'instruction_ids': instruction_ids
        }
        
    except Exception as e:
        # Return failure score on any error
        return {
            'score': 0.0,
            'acc': False,
            'error': str(e),
            'follow_all_instructions': False,
            'follow_instruction_list': [],
            'num_instructions': 0,
            'num_followed': 0,
            'accuracy': 0.0,
            'evaluation_mode': 'strict' if strict else 'loose'
        }

def compute_score_loose(solution_str: str, ground_truth: Union[str, Dict]) -> Dict[str, Any]:
    """
    Compute the IFEval instruction following score using loose evaluation.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: Either a JSON string or dict containing instruction information
        
    Returns:
        Dictionary containing the score and evaluation details
    """
    return compute_score(solution_str, ground_truth, strict=False) 