# Copyright 2024 Bytedance Ltd. and/or its affiliates

"""
Evaluation script for model performance on InfoBench benchmark.
Evaluates instruction following capabilities using decomposed question evaluation.

Takes a single JSONL input file with model responses and outputs evaluation results
to a JSONL file with added 'eval' field containing the scoring results.

Requires OpenAI API key for proper evaluation.
"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the specific module path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verl', 'utils', 'reward_score'))

# Import the InfoBench module directly
import infobench


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_single_entry(entry: Dict[str, Any], eval_model: str = "gpt-4.1") -> Dict[str, Any]:
    """Evaluate a single entry using InfoBench scoring."""
    # Extract the model's response text
    solution_str = entry['response']
    
    # Extract the ground truth (InfoBench evaluation data)
    ground_truth = {
        'decomposed_questions': entry['decomposed_questions'],
        'instruction': entry['instruction'],
        'id': entry['id'],
        'category': entry['category'],
        'subset': entry['subset']
    }
    
    # Compute score using InfoBench with OpenAI option
    result = infobench.compute_score(solution_str, ground_truth, eval_model=eval_model, return_verl_reward=False)
    
    # Add eval field to the original entry
    entry['eval'] = result['eval_results']
    
    return entry
    

def run_evaluation(input_path: str, output_path: str, max_workers: int = 50, eval_model: str = "gpt-4.1"):
    """
    Main function to run InfoBench evaluation on model outputs.
    
    Args:
        input_path: Path to input JSONL file with model responses
        output_path: Path to output JSONL file with evaluation results
        max_workers: Maximum number of concurrent workers
        eval_model: OpenAI model to use for evaluation
    """
    # Load input data
    data = load_jsonl(input_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Evaluating output from {input_path}")
    print(f"Using OpenAI model: {eval_model} with {max_workers} max workers")
    
    # Use ThreadPoolExecutor for concurrent evaluation
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_entry = {
            executor.submit(evaluate_single_entry, entry, eval_model): entry 
            for entry in data
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_entry), total=len(data), desc="Processing entries"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                entry = future_to_entry[future]
                print(f"Failed to evaluate entry {entry.get('id', 'unknown')}: {e}")
                # Create error result
                error_entry = entry.copy()
                error_entry['eval'] = [False] * len(entry.get('decomposed_questions', []))
                error_entry['error'] = str(e)
                results.append(error_entry)
    
    # Sort results by original order (by id if available, otherwise keep as is)
    id_to_index = {entry.get('id', i): i for i, entry in enumerate(data)}
    results.sort(key=lambda x: id_to_index.get(x.get('id'), len(data)))
    
    # Write results to output file
    with open(output_path, 'w', encoding='utf-8') as result_writer:
        for result in results:
            result_writer.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Calculate and print summary statistics
    print_summary_statistics(output_path)
    return output_path


def print_summary_statistics(output_path: str):
    """Print summary statistics from evaluation results."""
    data = load_jsonl(output_path)
    
    total_entries = len(data)
    total_questions = sum(len(entry.get('decomposed_questions', [])) for entry in data)
    
    # Calculate metrics
    perfect_scores = 0  # Entries with all questions correct
    total_correct = 0
    successful_evals = 0
    
    for entry in data:
        eval_results = entry.get('eval', [])
        
        if eval_results:
            successful_evals += 1
            correct_count = sum(1 for result in eval_results if result)
            total_correct += correct_count
            
            if correct_count == len(eval_results):
                perfect_scores += 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("INFOBENCH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total entries: {total_entries}")
    print(f"Successful evaluations: {successful_evals}")
    print(f"Total questions: {total_questions}")
    print(f"Perfect scores (all questions correct): {perfect_scores}")
    if successful_evals > 0:
        print(f"Prompt-level accuracy: {perfect_scores/successful_evals:.4f} ({perfect_scores}/{successful_evals})")
    if total_questions > 0:
        print(f"Question-level accuracy: {total_correct/total_questions:.4f} ({total_correct}/{total_questions})")
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs on InfoBench using decomposed questions with OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with OpenAI (requires OPENAI_API_KEY)
  python evaluate_infobench.py --input model_responses.jsonl --output results.jsonl
  
  # Loose evaluation mode with different OpenAI model
  python evaluate_infobench.py --input responses.jsonl --output results.jsonl --loose --eval_model gpt-4
  
  # Disable OpenAI evaluation (all questions will be marked as False)
  python evaluate_infobench.py --input responses.jsonl --output results.jsonl --no_openai

Note: This script requires OpenAI API access for proper evaluation. Without it, 
all questions will be marked as False.
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input JSONL file with model responses'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        required=True,
        help='Path to output JSONL file for evaluation results'
    )
    
    parser.add_argument(
        '--loose',
        action='store_true',
        help='Use loose evaluation mode instead of strict mode'
    )
    
    parser.add_argument(
        '--max_workers',
        type=int,
        default=50,
        help='Maximum number of concurrent workers (default: 50)'
    )
    
    parser.add_argument(
        '--eval_model',
        type=str,
        default='gpt-4.1',
        help='OpenAI model to use for evaluation (default: gpt-4.1)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist")
        sys.exit(1)
    
    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        max_workers=args.max_workers,
        eval_model=args.eval_model
    )
    
    print(f"\nEvaluation complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main() 