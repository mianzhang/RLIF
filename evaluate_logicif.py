# Copyright 2024 Bytedance Ltd. and/or its affiliates

"""
Evaluation script for model performance on LogicIF benchmark.
Uses fast rule-based JSON parsing (jsonparse) for structured data extraction and scoring.

Supports any model's responses in the LogicIF JSONL format and provides
comprehensive evaluation metrics including task-level and instance-level accuracy.
No external API calls required - evaluation runs completely locally.
"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Add the specific module path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verl', 'utils', 'reward_score'))

# Import the LogicIF module directly
import logicif


def load_model_responses(file_path: str) -> List[Dict[str, Any]]:
    """Load model responses from JSONL file."""
    responses = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        responses.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
        
        print(f"✅ Loaded {len(responses)} model responses from {file_path}")
        return responses
        
    except Exception as e:
        print(f"❌ Failed to load model responses: {e}")
        return []


def evaluate_single_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single model response using LogicIF scoring with fast rule-based parsing."""
    try:
        # Extract the model's response text (solution_str)
        solution_str = response['response']
        
        # Extract the ground truth (code_output field with task_id)
        task_id = f"{response['task_id']}-{response['test_case_id']}"
        ground_truth = {
            'task_id': task_id,
            'code_output': response['code_output']
        }
        
        # Compute score using LogicIF (rule-based JSON parsing)
        result = logicif.compute_score(solution_str, ground_truth, return_verl_reward=False)
        
        # Add additional info for analysis
        result['task_id'] = response['task_id']
        result['test_case_id'] = response['test_case_id']
        
        return result
        
    except Exception as e:
        task_id = response['task_id']
        test_case_id = response['test_case_id']
        return {
            'score': 0.0,
            'output_match': False,
            'stats_match': False,
            'both_match': False,
            'has_error': True,
            'error_message': str(e),
            'task_id': response['task_id'],
            'test_case_id': response['test_case_id'],
        }


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive statistics from evaluation results."""
    total_examples = len(results)
    
    if total_examples == 0:
        return {'error': 'No results to analyze'}
    
    # Basic metrics
    scores = [r['score'] for r in results]
    output_matches = [r['output_match'] for r in results]
    stats_matches = [r['stats_match'] for r in results]
    both_matches = [r['both_match'] for r in results]
    
    # Calculate averages
    avg_score = sum(scores) / total_examples
    output_accuracy = sum(output_matches) / total_examples
    stats_accuracy = sum(stats_matches) / total_examples
    both_accuracy = sum(both_matches) / total_examples
    
    # Error analysis
    error_count = sum(1 for r in results if r.get('has_error', False))
    success_count = total_examples - error_count
    
    # Task-level analysis
    task_results = {}
    for result in results:
        task_id = result['task_id']
        if task_id not in task_results:
            task_results[task_id] = []
        task_results[task_id].append(result)
    
    # Calculate task-level accuracies
    task_level_stats = {}
    for task_id, task_results_list in task_results.items():
        task_scores = [r['score'] for r in task_results_list]
        task_both_matches = [r['both_match'] for r in task_results_list]
        
        task_level_stats[task_id] = {
            'total_cases': len(task_results_list),
            'avg_score': sum(task_scores) / len(task_scores),
            'perfect_cases': sum(task_both_matches),
            'task_accuracy': sum(task_both_matches) / len(task_both_matches)
        }
    
    # Overall task-level accuracy (all test cases for a task must be correct)
    perfect_tasks = sum(1 for stats in task_level_stats.values() if stats['perfect_cases'] == stats['total_cases'])
    task_level_accuracy = perfect_tasks / len(task_level_stats)
    
    return {
        'total_examples': total_examples,
        'successful_evaluations': success_count,
        'failed_evaluations': error_count,
        'average_score': avg_score,
        'output_accuracy': output_accuracy,
        'stats_accuracy': stats_accuracy,
        'both_accuracy': both_accuracy,
        'task_level_accuracy': task_level_accuracy,
        'total_tasks': len(task_level_stats),
        'perfect_tasks': perfect_tasks,
        'task_breakdown': task_level_stats
    }


def print_results(stats: Dict[str, Any], model_name: str = "Model"):
    """Print formatted evaluation results."""
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} Performance on LogicIF")
    print("=" * 80)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total Examples: {stats['total_examples']}")
    print(f"   Successful Evaluations: {stats['successful_evaluations']}")
    print(f"   Failed Evaluations: {stats['failed_evaluations']}")
    print(f"   Total Tasks: {stats['total_tasks']}")
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Average Score: {stats['average_score']:.4f}")
    print(f"   Output Accuracy: {stats['output_accuracy']:.4f} ({stats['output_accuracy']*100:.2f}%)")
    print(f"   Stats Accuracy: {stats['stats_accuracy']:.4f} ({stats['stats_accuracy']*100:.2f}%)")
    print(f"   Both Match Accuracy: {stats['both_accuracy']:.4f} ({stats['both_accuracy']*100:.2f}%)")
    
    print(f"\n📋 Task-Level Analysis:")
    print(f"   Task-Level Accuracy: {stats['task_level_accuracy']:.4f} ({stats['task_level_accuracy']*100:.2f}%)")
    print(f"   Perfect Tasks: {stats['perfect_tasks']}/{stats['total_tasks']}")
    
    # Show top and bottom performing tasks
    task_breakdown = stats['task_breakdown']
    sorted_tasks = sorted(task_breakdown.items(), key=lambda x: x[1]['task_accuracy'], reverse=True)
    
    print(f"\n🏆 Top 5 Performing Tasks:")
    for i, (task_id, task_stats) in enumerate(sorted_tasks[:5]):
        accuracy = task_stats['task_accuracy']
        perfect = task_stats['perfect_cases']
        total = task_stats['total_cases']
        print(f"   {i+1}. {task_id}: {accuracy:.4f} ({perfect}/{total})")
    
    print(f"\n📉 Bottom 5 Performing Tasks:")
    for i, (task_id, task_stats) in enumerate(sorted_tasks[-5:]):
        accuracy = task_stats['task_accuracy']
        perfect = task_stats['perfect_cases']
        total = task_stats['total_cases']
        print(f"   {i+1}. {task_id}: {accuracy:.4f} ({perfect}/{total})")


def save_detailed_results(results: List[Dict[str, Any]], stats: Dict[str, Any], 
                         output_file: str, model_name: str = 'unknown_model'):
    """Save detailed results to JSON file."""
    detailed_results = {
        'evaluation_summary': stats,
        'individual_results': results,
        'evaluation_settings': {
            'model_evaluated': model_name,
            'evaluation_framework': 'logicif',
            'parsing_method': 'rule-based (jsonparse)'
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on LogicIF using fast, local rule-based JSON parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (no file output)
  python evaluate_logicif.py
  
  # Save results to file
  python evaluate_logicif.py --output_file results.json
  
  # Specify custom input file
  python evaluate_logicif.py --input_file model_responses.jsonl
  
  # Evaluate subset of responses and save results
  python evaluate_logicif.py --max_examples 100 --output_file results.json
  
  # Quick test in quiet mode
  python evaluate_logicif.py --max_examples 10 --quiet
  
  # Use custom model name and save to file
  python evaluate_logicif.py --input_file responses.jsonl --model_name my_model --output_file results.json
        """
    )
    
    parser.add_argument(
        '--input_file', '-i',
        type=str,
        default='./benchmark/logicifevalmini_sample_output.jsonl',
        help='Path to the JSONL file containing model responses'
    )
    
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        default=None,
        help='Path to save detailed evaluation results (optional, no file saved if not specified)'
    )
    
    parser.add_argument(
        '--max_examples', '-n',
        type=int,
        default=None,
        help='Maximum number of examples to evaluate (default: all examples)'
    )
    
    parser.add_argument(
        '--model_name', '-m',
        type=str,
        default=None,
        help='Name of the evaluated model (auto-detected from input file if not specified)'
    )
    
    parser.add_argument(
        '--reward_score_path',
        type=str,
        default='./verl/utils/reward_score',
        help='Path to the reward_score module directory'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function using fast, local rule-based JSON parsing."""
    args = parse_arguments()
    
    # Determine model name from input file if not specified
    if args.model_name is None:
        basename = os.path.basename(args.input_file)
        model_name = basename.split('-')[0] if '-' in basename else 'unknown'
    else:
        model_name = args.model_name
    
    if not args.quiet:
        print(f"{model_name.upper()} LogicIF Evaluation")
        print("=" * 50)
        print(f"📁 Input file: {args.input_file}")
        if args.output_file:
            print(f"💾 Output file: {args.output_file}")
        if args.max_examples:
            print(f"📊 Max examples: {args.max_examples}")
    
    # Update sys.path for reward_score module
    if args.reward_score_path not in sys.path:
        sys.path.insert(0, args.reward_score_path)
    
    # Load model responses
    responses = load_model_responses(args.input_file)
    
    if not responses:
        print("❌ No responses to evaluate")
        sys.exit(1)
    
    # Limit examples if specified
    if args.max_examples and args.max_examples < len(responses):
        responses = responses[:args.max_examples]
        if not args.quiet:
            print(f"🔢 Limited to first {args.max_examples} examples")
    
    if not args.quiet:
        print(f"\n🔄 Evaluating {len(responses)} responses using rule-based parsing...")
    
    # Evaluate all responses sequentially
    results = []
    failed_count = 0
    
    # Simple sequential evaluation with progress bar
    for response in tqdm(responses, desc="Evaluating responses", disable=args.quiet):
        result = evaluate_single_response(response)
        results.append(result)
        
        if result.get('has_error', False):
            failed_count += 1
    
    if not args.quiet:
        print(f"\n✅ Evaluation completed!")
        print(f"   Processed: {len(results)} responses")
        print(f"   Evaluation errors: {failed_count}")
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print results (unless quiet mode)
    if not args.quiet:
        print_results(stats, model_name)
    
    # Save detailed results if output file is specified
    if args.output_file:
        save_detailed_results(results, stats, args.output_file, model_name)
    
    # Print summary
    print(f"\n🎉 Evaluation complete!")
    print(f"📈 {model_name.upper()} achieved {stats['both_accuracy']*100:.2f}% accuracy on LogicIF")
    print(f"📊 Task-level accuracy: {stats['task_level_accuracy']*100:.2f}%")
    
    if args.quiet:
        # In quiet mode, print just the key metrics
        print(f"Results: {stats['both_accuracy']:.4f} accuracy, {stats['task_level_accuracy']:.4f} task-level accuracy")


if __name__ == "__main__":
    main() 