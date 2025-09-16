#!/usr/bin/env python3
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

"""InfoBench instruction following evaluation for VERL reward scoring."""

import json
import re
import os
import time
from typing import Dict, List, Any, Union, Optional
from openai import OpenAI

# System message for InfoBench evaluation (based on original evaluation.py)
INFOBENCH_SYS_MSG = """Based on the provided Instruction and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:

- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration, consider a question that asks, "Does each sentence in the generated text use a second person?" If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question.

- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks, "Is the second sentence in the generated text a compound sentence?" and the generated text only has one sentence, it offers no relevant information to answer the question. Consequently, the answer should be 'NO'."""


def extract_yes_no_answer(response: str) -> Optional[bool]:
    """
    Extract YES/NO answer from LLM response.
    
    Args:
        response: The LLM response text
        
    Returns:
        True for YES, False for NO, None if unclear
    """
    response_lower = response.lower().strip()
    
    # Direct yes/no at the start
    if response_lower.startswith("yes"):
        return True
    elif response_lower.startswith("no"):
        return False
    
    # Check for YES/NO in uppercase
    if "YES" in response and "NO" not in response:
        return True
    elif "YES" not in response and "NO" in response:
        return False
    
    # Check for explicit patterns
    yes_patterns = [
        r'\byes\b',
        r'\bYES\b',
        r'answer is yes',
        r'answer: yes',
        r'the answer is yes'
    ]
    
    no_patterns = [
        r'\bno\b',
        r'\bNO\b', 
        r'answer is no',
        r'answer: no',
        r'the answer is no'
    ]
    
    yes_matches = sum(1 for pattern in yes_patterns if re.search(pattern, response, re.IGNORECASE))
    no_matches = sum(1 for pattern in no_patterns if re.search(pattern, response, re.IGNORECASE))
    
    if yes_matches > no_matches:
        return True
    elif no_matches > yes_matches:
        return False
    
    return None


def evaluate_single_question_with_openai(client: OpenAI, eval_model: str, input_task: str, 
                                        output: str, question: str, message_history: List = None, 
                                        max_retries: int = 3) -> tuple:
    """
    Evaluate a single question using OpenAI API.
    
    Args:
        client: OpenAI client
        eval_model: Model name for evaluation
        input_task: Original instruction
        output: Generated text to evaluate
        question: Question to ask about the output
        message_history: Previous messages in conversation (for multi-turn)
        max_retries: Maximum number of retries on failure
    
    Returns:
        tuple: (result, updated_message_history)
    """
    message = message_history.copy() if message_history else []
    
    if len(message) == 0:
        content = f"{INFOBENCH_SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
    else:
        content = f"{question}\n"
    
    message.append({"role": "user", "content": content})
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=eval_model,
                messages=message,
            )
            generation = completion.choices[0].message.content
            message.append({"role": "assistant", "content": generation})
            
            # Parse the response using the existing function
            result = extract_yes_no_answer(generation)
            
            if result is None:
                print(f"Ambiguous answer for question: {question}")
                print(f"Response: {generation}")
                # Default to False for ambiguous responses
                result = False
            
            return result, message
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(20)
            else:
                print(f"Failed to evaluate question after {max_retries} attempts")
                return None, message
    
    return None, message


def evaluate_decomposed_questions(solution_str: str, ground_truth: Dict[str, Any], 
                                 eval_model: str = "gpt-4.1") -> List[bool]:
    """
    Evaluate the solution against decomposed questions using OpenAI.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: Dictionary containing decomposed questions and other info
        eval_model: OpenAI model to use for evaluation
        
    Returns:
        List of boolean results for each question
    """
    decomposed_questions = ground_truth['decomposed_questions']
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. All questions will be marked as False.")
        return [False] * len(decomposed_questions)
    
    results = []
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        instruction = ground_truth['instruction']
        
        # Use sequential evaluation to maintain conversation context
        message_history = []
        for question in decomposed_questions:
            result, message_history = evaluate_single_question_with_openai(
                client, eval_model, instruction, solution_str, question, message_history
            )
            if result is not None:
                results.append(result)
            else:
                # If OpenAI evaluation fails, mark as False
                print(f"OpenAI evaluation failed for question: {question}, marking as False")
                results.append(False)
        
    except Exception as e:
        print(f"Error in OpenAI evaluation: {e}")
        # If there's an error, mark all questions as False
        results = [False] * len(decomposed_questions)
    
    return results


def compute_score(solution_str: str, ground_truth: Union[str, Dict], eval_model: str = "gpt-4.1", return_verl_reward: bool = True) -> Dict[str, Any]:
    """
    Compute InfoBench instruction following score using OpenAI evaluation.
    
    Args:
        solution_str: The model's response to evaluate
        ground_truth: JSON string or dict containing InfoBench evaluation data
        eval_model: OpenAI model to use for evaluation
        return_verl_reward: Whether to return only the score (for VERL) or full results
        
    Returns:
        Dictionary containing score and evaluation details
    """
    # Evaluate against decomposed questions
    eval_results = evaluate_decomposed_questions(solution_str, ground_truth, eval_model)
    
    # Calculate scores
    total_questions = len(eval_results)
    correct_answers = sum(1 for result in eval_results if result is True)
    
    if total_questions == 0:
        accuracy = 0.0
    else:
        accuracy = correct_answers / total_questions
    
    # Convert to reward score: 1.0 for perfect, 0.0 otherwise
    score = 1.0 if accuracy == 1.0 else 0.0
    
    if return_verl_reward:
        return {
            "score": score,
        } 
    else:
        return {
            "score": score,
            "eval_results": eval_results,
        }
