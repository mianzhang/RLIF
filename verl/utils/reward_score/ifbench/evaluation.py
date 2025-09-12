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

"""Core evaluation functions for IFBench instruction following."""

import json
from typing import Dict, List, Any, Union
import re

from .utils import import_ifbench_modules, MockInputExample


def remove_think_tags(text: str) -> str:
    """
    Remove <think>...</think> tags from the text.
    
    Args:
        text: Input text that may contain <think>...</think> tags
        
    Returns:
        Text with <think>...</think> sections removed
    """
    # Use regex to remove <think>...</think> tags and their content
    # This handles multiline content and is case-insensitive
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace that might be left
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Replace multiple newlines with double newlines
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


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
    instructions_registry, evaluation_lib = import_ifbench_modules()
    
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


def compute_score_internal(solution_str: str, ground_truth: Union[str, Dict], strict: bool = True, return_verl_reward: bool = True) -> Dict[str, Any]:
    """
    Internal function to compute the IFBench instruction following score.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: Either a JSON string or dict containing instruction information
        strict: Whether to use strict evaluation mode
        
    Returns:
        Dictionary containing the score and evaluation details
    """
    try:
        # Remove <think>...</think> tags from the solution
        cleaned_solution = remove_think_tags(solution_str)
        
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
                'follow_all_instructions': True,
                'follow_instruction_list': [],
                'num_instructions': 0,
                'num_followed': 0,
                'accuracy': 1.0,
                'evaluation_mode': 'strict' if strict else 'loose'
            }
        
        # Evaluate instruction following using cleaned solution
        eval_result = evaluate_instruction_following(
            response=cleaned_solution,
            instruction_ids=instruction_ids,
            kwargs_list=kwargs_list,
            prompt=prompt,
            strict=strict
        )
        
        # Calculate score
        # Use binary score: 1.0 if all instructions followed, 0.0 otherwise
        score = 1.0 if eval_result['follow_all_instructions'] else 0.0
        
        if return_verl_reward:
            return {
                'score': score,
                'num_instructions': eval_result['num_instructions'],
                'num_followed': eval_result['num_followed'],
                'has_error': False
            }
        else:
            return {
                'score': score,
                'follow_all_instructions': eval_result['follow_all_instructions'],
                'follow_instruction_list': eval_result['follow_instruction_list'],
                'num_instructions': eval_result['num_instructions'],
                'num_followed': eval_result['num_followed'],
                'accuracy': eval_result['accuracy'],
                'evaluation_mode': 'strict' if strict else 'loose',
                'instruction_ids': instruction_ids,
                'has_error': False
            }
        
    except Exception as e:
        # Return failure score on any error
        if return_verl_reward:
            return {
                'score': 0.0,
                'num_instructions': 0,
                'num_followed': 0,
                'has_error': True
            }
        else:
            return {
                'score': 0.0,
                'error': str(e),
                'follow_all_instructions': False,
                'follow_instruction_list': [],
                'num_instructions': 0,
                'num_followed': 0,
                'accuracy': 0.0,
                'evaluation_mode': 'strict' if strict else 'loose',
                'has_error': True
            } 