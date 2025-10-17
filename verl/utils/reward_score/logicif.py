# Copyright 2024 Bytedance Ltd. and/or its affiliates

"""
LogicIF reward scoring module for VERL.

This module evaluates whether model responses correctly execute algorithmic tasks
by comparing both the final output and execution statistics against ground truth.
LogicIF focuses on manual function execution and algorithmic reasoning.
Uses jsonparse for structured extraction from model responses.
"""

import json
import re
import random
from typing import Dict, Any, Union, Tuple, List
import jsonparse


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


def extract_output_with_jsonparse(response: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Extract structured output and statistics from LLM response using jsonparse.
    Looks for JSON objects in the response and validates they have required fields.
    
    Args:
        response: LLM response to extract from
        
    Returns:
        Tuple of (output, stats)
    """
    # Try to extract the last JSON object from the response
    json_obj = jsonparse.extract_valid_json(response, position='last')
    
    if json_obj is None:
        return None, {}
    
    # Validate the JSON has required fields
    if isinstance(json_obj, dict):
        # Check if it has 'output' and 'stats' fields
        if 'output' in json_obj and 'stats' in json_obj:
            output = json_obj.get('output', None)
            stats = json_obj.get('stats', {})
            return output, stats
    
    # If we get here, the JSON wasn't in the expected format
    return None, {}
        

def parse_model_response(response: str, expected_stats_keys: List[str] = None) -> Dict[str, Any]:
    """
    Parse the model's response using jsonparse to extract the output and statistics.
    
    Args:
        response: The model's response text
        expected_stats_keys: List of expected statistics keys (unused, kept for compatibility)
        
    Returns:
        Dictionary with parsed 'output' and 'stats', or None values if parsing fails
    """
    # Use jsonparse to extract structured data
    output, stats = extract_output_with_jsonparse(response)
    
    return {'output': output, 'stats': stats}


def compare_outputs(expected: Any, actual: Any) -> bool:
    """
    Compare expected and actual outputs.
    
    Args:
        expected: Expected output value
        actual: Actual output value from model
        
    Returns:
        True if outputs match, False otherwise
    """
    if expected is None or actual is None:
        return expected == actual
    
    # Handle numeric comparisons with some tolerance for floating point
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 1e-9
    
    # Direct equality for other types
    return expected == actual


def compare_stats(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """
    Compare expected and actual statistics dictionaries.
    
    Args:
        expected: Expected stats dictionary
        actual: Actual stats dictionary from model
        
    Returns:
        True if stats match exactly, False otherwise
    """
    if expected is None or actual is None:
        return False
    
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    
    # Check if all expected keys are present with correct values
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        
        actual_value = actual[key]
        
        # Handle boolean comparisons
        if isinstance(expected_value, bool) and isinstance(actual_value, bool):
            if expected_value != actual_value:
                return False
        # Handle numeric comparisons
        elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(expected_value - actual_value) > 1e-9:
                return False
        # Handle exact equality for other types
        else:
            if expected_value != actual_value:
                return False
    
    # Check if actual has extra keys (should match exactly)
    if set(expected.keys()) != set(actual.keys()):
        return False
    
    return True


def compute_score(solution_str: str, ground_truth: Union[str, Dict[str, Any]], 
                 return_verl_reward: bool = True) -> Dict[str, Any]:
    """
    Compute LogicIF reward score by comparing model output with expected results.
    
    Args:
        solution_str: The model's response text
        ground_truth: Ground truth data (JSON string or dict) containing:
                     - task_id: Task identifier string
                     - code_output: Dict with 'output' and 'stats' fields
        return_verl_reward: If True, return minimal dict with just score
    Returns:
        Dictionary containing:
        - score: Binary score (1.0 for success, 0.0 for failure)
        - output_match: Whether outputs match
        - stats_match: Whether stats match  
        - both_match: Whether both output and stats match
        - expected_output: The expected output value
        - actual_output: The parsed actual output value
        - expected_stats: The expected stats dictionary
        - actual_stats: The parsed actual stats dictionary
        - task_id: The task identifier
    """
    # Remove <think>...</think> tags from the solution
    cleaned_solution = remove_think_tags(solution_str)
    
    # Parse ground truth
    if isinstance(ground_truth, str):
        gt_data = json.loads(ground_truth)
    else:
        gt_data = ground_truth
        
    code_output = gt_data['code_output']
    expected_output = code_output['output']
    expected_stats = code_output['stats']
        
    # Parse model response using jsonparse with cleaned solution
    parsed_response = parse_model_response(cleaned_solution)
    actual_output = parsed_response['output']
    actual_stats = parsed_response['stats']
        
    output_match = compare_outputs(expected_output, actual_output)
    stats_match = compare_stats(expected_stats, actual_stats)
    both_match = output_match and stats_match
    
    # Binary score: 1.0 if both match, 0.0 otherwise
    score = 1.0 if both_match else 0.0
    
    if random.random() < 0.05:
        print(f"Score: {score}")
        print(f"Output Match: {output_match}")
        print(f"Stats Match: {stats_match}")
        print(f"Both Match: {both_match}")
        print(f"Expected Output: {expected_output}")
        print(f"Actual Output: {actual_output}")
        print(f"Expected Stats: {expected_stats}")
        print(f"Solution: {solution_str}")
        print(f"Cleaned Solution: {cleaned_solution}")

    if return_verl_reward:
        return {
            'score': score,
            'has_error': False
        }
    else:
        return {
            'score': score,
            'output_match': output_match,
            'stats_match': stats_match,
            'both_match': both_match,
            'expected_output': expected_output,
            'actual_output': actual_output,
            'expected_stats': expected_stats,
            'actual_stats': actual_stats,
        }
