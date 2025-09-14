"""
LogicIF Mini reward scoring module for VERL.

This module evaluates whether model responses correctly execute algorithmic tasks
by comparing both the final output and execution statistics against ground truth.
LogicIF focuses on manual function execution and algorithmic reasoning.
Uses OpenAI models (default: gpt-5-mini) for structured extraction.
"""

import json
import time
import re
import random
from typing import Dict, Any, Union, Optional, Tuple, List


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


def analyze_output_format(output_value):
    """
    Analyze the format/structure of an output value and return a string description.
    Adapted from RLIF_data/LogicIF/evaluation.py
    
    Args:
        output_value: The output value to analyze
        
    Returns:
        String description of the format
    """
    if output_value is None:
        return "None"
    elif isinstance(output_value, bool):
        return "bool"
    elif isinstance(output_value, int):
        return "int"
    elif isinstance(output_value, float):
        return "float"
    elif isinstance(output_value, str):
        return "str"
    elif isinstance(output_value, list):
        if len(output_value) == 0:
            return "list[]"
        
        # Check if all elements are the same type
        first_type = type(output_value[0])
        if all(isinstance(item, first_type) for item in output_value):
            if first_type == int:
                return "list[int]"
            elif first_type == float:
                return "list[float]"
            elif first_type == str:
                return "list[str]"
            elif first_type == bool:
                return "list[bool]"
            elif first_type == list:
                # Nested list - analyze the structure of first element
                inner_format = analyze_output_format(output_value[0])
                return f"list[{inner_format}]"
            else:
                return f"list[{first_type.__name__}]"
        else:
            # Mixed types - show the types of first few elements
            types_seen = [type(item).__name__ for item in output_value[:3]]
            if len(output_value) > 3:
                return f"list[mixed: {', '.join(types_seen)}, ...]"
            else:
                return f"list[mixed: {', '.join(types_seen)}]"
    
    elif isinstance(output_value, tuple):
        if len(output_value) == 0:
            return "tuple()"
        
        # Analyze each element of the tuple
        element_formats = []
        for item in output_value:
            element_formats.append(analyze_output_format(item))
        
        return f"tuple({', '.join(element_formats)})"
    
    elif isinstance(output_value, dict):
        if len(output_value) == 0:
            return "dict{}"
        
        # Analyze a few key-value pairs
        sample_items = list(output_value.items())[:3]
        key_types = set(type(k).__name__ for k, v in sample_items)
        value_types = set(type(v).__name__ for k, v in sample_items)
        
        if len(key_types) == 1 and len(value_types) == 1:
            return f"dict[{list(key_types)[0]}: {list(value_types)[0]}]"
        else:
            return f"dict[mixed keys/values]"
    
    elif isinstance(output_value, set):
        if len(output_value) == 0:
            return "set()"
        
        # Check element types
        first_item = next(iter(output_value))
        first_type = type(first_item)
        if all(isinstance(item, first_type) for item in output_value):
            return f"set[{first_type.__name__}]"
        else:
            return "set[mixed]"
    
    else:
        # Unknown/custom type
        return f"{type(output_value).__name__}"


# Extraction prompt template (adapted from LogicIF)
EXTRACT_OUTPUT_PROMPT = """You are a data extraction expert. Parse the algorithm execution response and extract structured output and statistics.

**TASK**: Extract the main result and statistics from the manual execution response.

**Context:**
Expected Statistics Keys: {stats_keys}
Expected Output Format: {output_format}

**Response to Parse:**
{llm_response}

**EXTRACTION RULES:**
1. **Output Field**: Extract ONLY the main result value
   - NOT the tuple (result, stats)
   - NOT the statistics dictionary
   - Just the primary return value following the expected format: {output_format}
   - Ensure the output matches the expected format exactly
   - Convert data types as needed (e.g., ensure lists are lists, integers are integers)

2. **Stats Field**: Extract statistics as a dictionary
   - Use the expected statistics keys: {stats_keys}
   - Values must be integers or booleans as appropriate
   - If missing, infer from reasoning or set to appropriate default values

**EXAMPLES:**
If response says "**Output:** 42" and "**Statistics:** {{"operations": 10, "flag": true}}":
- Output: 42
- Stats: {{"operations": 10, "flag": true}}

If response mentions "Final result is [1, 2, 3] with 5 comparisons and 3 swaps":
- Output: [1, 2, 3]
- Stats: {{"comparisons": 5, "swaps": 3}}

Return JSON format:
{{
    "output": extracted_main_result,
    "stats": {{"stat_key": value}}
}}"""


def openai_inference(conversations: Union[List[List[Dict]], List[Dict]], 
                     model: str = "gpt-5-nano", 
                     return_json: bool = False, 
                     temperature: Optional[float] = None) -> List[str]:
    """
    OpenAI inference function for extraction.
    Simplified version of LogicIF's openai_inference.
    """
    from openai import OpenAI
    
    if not isinstance(conversations, list):
        conversations = [conversations]
    
    ret = []
    client = OpenAI()
    
    for conv in conversations:
        try:
            kwargs = dict(
                model=model,
                messages=conv,
                response_format={"type": "json_object"} if return_json else None,
            )
            
            # Add temperature if provided and not a reasoning model
            if temperature is not None and not any(key in model.lower() for key in ["o3", "o1", "o4-mini", "gpt-5"]):
                kwargs["temperature"] = temperature

            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            completion = client.chat.completions.create(**kwargs)
            generation = completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error during OpenAI inference: {str(e)}")
            print(f"Completion: {completion}")
            generation = '[ERROR]'
        
        ret.append(generation)
        
    return ret


def extract_output_with_openai(response: str, stats_keys: List[str], 
                               extract_model: str = "gpt-5-nano", 
                               output_format: str = "unknown") -> Tuple[Any, Dict[str, Any], str]:
    """
    Extract structured output and statistics from LLM response using OpenAI.
    Adapted from LogicIF's extract_output function.
    
    Args:
        response: LLM response to extract from
        stats_keys: Expected statistics keys
        extract_model: Model to use for extraction (e.g., 'gpt-5-nano')
        output_format: Expected output format
        
    Returns:
        Tuple of (output, stats, extract_response)
    """
    def gen_forward():
        try:
            conv = [
                {"role": "user", "content": EXTRACT_OUTPUT_PROMPT.format(
                    llm_response=response, 
                    stats_keys=stats_keys,
                    output_format=output_format
                )},
            ]
            extract_response = openai_inference([conv], model=extract_model, return_json=True)[0]
            
            # Check if the extraction response is "[ERROR]"
            if extract_response == "[ERROR]":
                return {}, "[ERROR]"
                
            json_obj = json.loads(extract_response)
            return json_obj, extract_response
        except json.JSONDecodeError as e:
            print(f"JSON decode error in extract_output: {e}", flush=True)
            if 'extract_response' in locals():
                print(f"Raw response: {extract_response[:200]}...", flush=True)
            # Return empty dict to trigger retry
            return {}, extract_response if 'extract_response' in locals() else ""
        except Exception as e:
            print(f"Unexpected error in extract_output: {e}", flush=True)
            return {}, ""

    max_retries = 5
    base_delay = 1.0
    retry_count = 0
    
    while retry_count < max_retries:
        json_obj, extract_response = gen_forward()
        
        # Check if we got "[ERROR]" response and retry
        if extract_response == "[ERROR]":
            if retry_count < max_retries - 1:
                delay = base_delay * (2 ** retry_count)  # Exponential backoff
                time.sleep(delay)
                retry_count += 1
                continue
            else:
                # Final attempt failed with "[ERROR]"
                return None, {}, "[ERROR]"
        
        # Check if we have required fields
        if isinstance(json_obj, dict) and 'output' in json_obj and 'stats' in json_obj:
            output = json_obj.get('output', None)
            stats = json_obj.get('stats', {})
            return output, stats, extract_response
        
        # Missing required fields, retry
        if retry_count < max_retries - 1:
            retry_count += 1
            delay = base_delay * (2 ** (retry_count - 1))
            time.sleep(delay)
            continue
        else:
            break
    
    # All retries failed, return what we have
    output = json_obj.get('output', None) if isinstance(json_obj, dict) else None
    stats = json_obj.get('stats', {}) if isinstance(json_obj, dict) else {}
    
    return output, stats, extract_response


def parse_model_response(response: str, expected_stats_keys: List[str] = None, 
                        extract_model: str = "gpt-5-nano", 
                        expected_output: Any = None) -> Dict[str, Any]:
    """
    Parse the model's response using OpenAI to extract the output and statistics.
    
    Args:
        response: The model's response text
        expected_stats_keys: List of expected statistics keys
        extract_model: OpenAI model to use for extraction
        expected_output: Expected output value for format analysis
        
    Returns:
        Dictionary with parsed 'output' and 'stats', or None values if parsing fails
    """
    if expected_stats_keys is None:
        expected_stats_keys = []
    
    try:
        # Analyze expected output format
        output_format = analyze_output_format(expected_output) if expected_output is not None else "unknown"
        
        # Use OpenAI to extract structured data
        output, stats, extract_response = extract_output_with_openai(
            response, expected_stats_keys, extract_model, output_format
        )
        
        if extract_response == "[ERROR]":
            print(f"Error in OpenAI-based parsing: {extract_response}")
            return {'output': None, 'stats': None}
        
        return {'output': output, 'stats': stats}
        
    except Exception as e:
        print(f"Error in OpenAI-based parsing: {e}")
        return {'output': None, 'stats': None}


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
                 extract_model: str = "gpt-5-nano", return_verl_reward: bool = True) -> Dict[str, Any]:
    """
    Compute LogicIF reward score by comparing model output with expected results.
    
    Args:
        solution_str: The model's response text
        ground_truth: Ground truth data (JSON string or dict) containing:
                     - task_id: Task identifier string
                     - code_output: Dict with 'output' and 'stats' fields
        extract_model: OpenAI model to use for extraction (default: "gpt-5-nano")
        
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
        - has_error: Whether there was an error during evaluation
        - error_message: Error details if has_error is True
    """
    print("Compute score for LogicIF Mini!!!!!!!") 
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
        
    # Parse model response using OpenAI with cleaned solution
    try:
        expected_stats_keys = list(expected_stats.keys()) if expected_stats else []
        parsed_response = parse_model_response(cleaned_solution, expected_stats_keys, extract_model, expected_output)
        
        actual_output = parsed_response['output']
        actual_stats = parsed_response['stats']
        
    except Exception as e:
        if return_verl_reward:
            return {
                'score': 0.0,
                'has_error': True
            }
        else:
            return {
                'score': 0.0,
                'output_match': False,
                'stats_match': False,
                'both_match': False,
                'expected_output': expected_output,
                'actual_output': None,
                'expected_stats': expected_stats,
                'actual_stats': None,
                'has_error': True,
                'error_message': f'Failed to parse model response: {str(e)}'
            }
    
    # Compare outputs and stats
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
            'has_error': False,
            'error_message': None
        }


# Convenience function for testing
def compute_score_with_details(solution_str: str, ground_truth: Union[str, Dict[str, Any]], 
                              extract_model: str = "gpt-5-nano") -> Dict[str, Any]:
    """
    Compute score with additional debugging details.
    Same as compute_score but includes extra information for analysis.
    """
    result = compute_score(solution_str, ground_truth, extract_model=extract_model)
    
    # Add additional debugging info
    result['evaluation_type'] = 'logicifevalmini'
    result['requires_both_match'] = True
    result['extraction_method'] = 'openai'
    
    return result 