# Copyright 2024 Bytedance Ltd. and/or its affiliates

"""
Unified evaluation script for instruction following benchmarks (IFBench and IFEval).
Evaluates instruction following capabilities using both strict and loose evaluation modes.

Supports any model's responses in the IFBench or IFEval JSONL format and provides
comprehensive evaluation metrics including prompt-level and instruction-level accuracy.
"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Add the specific module path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verl', 'utils', 'reward_score'))


def load_jsonl_file(file_path: str, file_type: str = "data") -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {file_type} file: {e}")
                        continue
        
        print(f"✅ Loaded {len(data)} entries from {file_type} file: {file_path}")
        return data
        
    except Exception as e:
        print(f"❌ Failed to load {file_type} file: {e}")
        return []


def create_prompt_to_response_dict(responses: List[Dict[str, Any]]) -> Dict[str, str]:
    """Create mapping from prompt to response."""
    prompt_to_response = {}
    for item in responses:
        prompt = item.get('prompt', '')
        response = item.get('response', '')
        if prompt and response:
            prompt_to_response[prompt] = response
    return prompt_to_response


def match_ground_truth_with_responses(ground_truth_data: List[Dict[str, Any]], 
                                    prompt_to_response: Dict[str, str]) -> List[Dict[str, Any]]:
    """Match ground truth data with model responses by prompt."""
    matched_data = []

    for gt_item in ground_truth_data:
        prompt = gt_item.get('prompt', '')
        if prompt in prompt_to_response:
            # Create combined data structure
            combined_item = {
                'prompt': prompt,
                'response': prompt_to_response[prompt],
                'instruction_ids': gt_item.get('instruction_id_list', []),
                'kwargs': gt_item.get('kwargs', []),
                'key': gt_item.get('key', '')
            }
            matched_data.append(combined_item)
        else:
            print(f"Warning: No response found for prompt: {prompt[:50]}...")
    
    print(f"✅ Matched {len(matched_data)} ground truth items with responses")
    return matched_data


def evaluate_single_response(response: Dict[str, Any], framework: str, strict: bool = True) -> Dict[str, Any]:
    """Evaluate a single model response using the specified framework."""
    try:
        # Import the appropriate framework module
        if framework == 'ifbench':
            from ifbench import compute_score as ifbench_compute_score
            compute_func = ifbench_compute_score
        elif framework == 'ifeval':
            from ifeval import compute_score as ifeval_compute_score
            compute_func = ifeval_compute_score
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Extract the model's response text
        solution_str = response.get('response', '')
        
        # Extract the ground truth (instruction information)
        ground_truth = {
            'instruction_ids': response.get('instruction_ids', []),
            'kwargs': response.get('kwargs', []),
            'prompt': response.get('prompt', '')
        }
        
        # Compute score using the specified framework
        result = compute_func(solution_str, ground_truth, strict=strict, return_verl_reward=False)
        
        # Add additional info for analysis
        result['prompt'] = response.get('prompt', '')
        result['response'] = solution_str
        result['model_used'] = response.get('model_used', 'unknown')
        result['framework'] = framework
        
        return result
        
    except Exception as e:
        print(f"Error evaluating response for prompt: {response.get('prompt', 'unknown')[:50]}...: {e}")
        return {
            'score': 0.0,
            'acc': False,
            'follow_all_instructions': False,
            'follow_instruction_list': [],
            'num_instructions': len(response.get('instruction_ids', [])),
            'num_followed': 0,
            'accuracy': 0.0,
            'has_error': True,
            'error_message': str(e),
            'prompt': response.get('prompt', ''),
            'response': response.get('response', ''),
            'model_used': response.get('model_used', 'unknown'),
            'evaluation_mode': 'strict' if strict else 'loose',
            'framework': framework
        }


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive statistics from evaluation results."""
    total_examples = len(results)
    
    if total_examples == 0:
        return {'error': 'No results to analyze'}
    
    # Basic metrics
    scores = [r['score'] for r in results]
    follow_all = [r['follow_all_instructions'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Calculate averages
    avg_score = sum(scores) / total_examples
    prompt_level_accuracy = sum(follow_all) / total_examples
    instruction_level_accuracy = sum(accuracies) / total_examples
    
    # Error analysis
    error_count = sum(1 for r in results if r.get('has_error', False))
    success_count = total_examples - error_count
    
    # Instruction-level analysis
    total_instructions = sum(r['num_instructions'] for r in results)
    total_followed = sum(r['num_followed'] for r in results)
    
    # Distribution analysis
    instruction_counts = {}
    for result in results:
        num_inst = result['num_instructions']
        if num_inst not in instruction_counts:
            instruction_counts[num_inst] = {'total': 0, 'perfect': 0}
        instruction_counts[num_inst]['total'] += 1
        if result['follow_all_instructions']:
            instruction_counts[num_inst]['perfect'] += 1
    
    return {
        'total_examples': total_examples,
        'successful_evaluations': success_count,
        'failed_evaluations': error_count,
        'average_score': avg_score,
        'prompt_level_accuracy': prompt_level_accuracy,
        'instruction_level_accuracy': instruction_level_accuracy,
        'total_instructions': total_instructions,
        'total_followed': total_followed,
        'instruction_counts': instruction_counts,
        'evaluation_mode': results[0].get('evaluation_mode', 'strict') if results else 'unknown',
        'framework': results[0].get('framework', 'unknown') if results else 'unknown'
    }


def print_results(stats: Dict[str, Any], model_name: str):
    """Print formatted evaluation results."""
    framework_name = stats['framework'].upper()
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} Performance on {framework_name} ({stats['evaluation_mode'].title()} Mode)")
    print("=" * 80)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total Examples: {stats['total_examples']}")
    print(f"   Successful Evaluations: {stats['successful_evaluations']}")
    print(f"   Failed Evaluations: {stats['failed_evaluations']}")
    print(f"   Total Instructions: {stats['total_instructions']}")
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Average Score: {stats['average_score']:.4f}")
    print(f"   Prompt-Level Accuracy: {stats['prompt_level_accuracy']:.4f} ({stats['prompt_level_accuracy']*100:.2f}%)")
    inst_accuracy = stats['instruction_level_accuracy']
    print(f"   Instruction-Level Accuracy: {inst_accuracy:.4f} ({inst_accuracy*100:.2f}%)")
    print(f"   Instructions Followed: {stats['total_followed']}/{stats['total_instructions']}")
    
    # Show distribution by instruction count
    print(f"\n📋 Performance by Instruction Count:")
    instruction_counts = stats['instruction_counts']
    for num_inst in sorted(instruction_counts.keys()):
        data = instruction_counts[num_inst]
        accuracy = data['perfect'] / data['total'] if data['total'] > 0 else 0
        print(f"   {num_inst} instructions: {accuracy:.4f} ({data['perfect']}/{data['total']})")


def save_detailed_results(results: List[Dict[str, Any]], stats: Dict[str, Any], 
                         output_file: str, model_name: str = 'unknown', 
                         evaluation_mode: str = 'strict', framework: str = 'unknown'):
    """Save detailed results to JSON file."""
    detailed_results = {
        'evaluation_summary': stats,
        'individual_results': results,
        'evaluation_settings': {
            'model_evaluated': model_name,
            'evaluation_framework': framework,
            'evaluation_mode': evaluation_mode
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
        description="Evaluate model performance on instruction following benchmarks (IFBench/IFEval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using IFBench framework with default files (strict mode)
  python evaluate_if.py --framework ifbench
  
  # Evaluate using IFEval framework with custom files (loose mode)
  python evaluate_if.py --framework ifeval --input_file responses.jsonl --ground_truth_file ground_truth.jsonl --loose
  
  # Use default files for framework
  python evaluate_if.py --framework ifbench
  python evaluate_if.py --framework ifeval
  
  # Evaluate subset of responses in quiet mode
  python evaluate_if.py --framework ifeval --max_examples 100 --quiet
        """
    )
    
    parser.add_argument(
        '--framework', '-f',
        type=str,
        choices=['ifbench', 'ifeval'],
        required=True,
        help='Evaluation framework to use (ifbench or ifeval).'
    )
    
    parser.add_argument(
        '--input_file', '-i',
        type=str,
        default=None,
        help='Path to the JSONL file containing model responses'
    )
    
    parser.add_argument(
        '--ground_truth_file', '-g',
        type=str,
        default=None,
        help='Path to the JSONL file containing ground truth data (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        default=None,
        help='Path to save detailed evaluation results (auto-generated if not specified)'
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
        '--loose',
        action='store_true',
        help='Use loose evaluation mode instead of strict mode'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    return parser.parse_args()


def get_default_paths(framework: str):
    """Get default file paths for each framework."""
    defaults = {
        'ifbench': {
            'response_file': './benchmark/ifbench_sample_output.jsonl',
            'ground_truth_file': './benchmark/ifbench.jsonl'
        },
        'ifeval': {
            'response_file': './benchmark/ifeval_sample_output.jsonl',
            'ground_truth_file': './benchmark/ifeval.jsonl'
        }
    }
    return defaults.get(framework, defaults['ifbench'])


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Get default paths for the framework
    default_paths = get_default_paths(args.framework)
    
    # Set default input files if not specified
    if args.input_file is None:
        args.input_file = default_paths['response_file']
    if args.ground_truth_file is None:
        args.ground_truth_file = default_paths['ground_truth_file']
    
    # Determine model name from input file if not specified
    if args.model_name is None:
        basename = os.path.basename(args.input_file)
        model_name = basename.split('.')[0] if '.' in basename else 'unknown'
    else:
        model_name = args.model_name
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = f'./{args.framework}_evaluation_results.json'
    
    evaluation_mode = 'loose' if args.loose else 'strict'
    
    if not args.quiet:
        print(f"{model_name.upper()} {args.framework.upper()} Evaluation ({evaluation_mode.title()} Mode)")
        print("=" * 70)
        print(f"🔧 Framework: {args.framework}")
        print(f"📁 Response file: {args.input_file}")
        print(f"📋 Ground truth file: {args.ground_truth_file}")
        print(f"💾 Output file: {args.output_file}")
        print(f"🔍 Evaluation mode: {evaluation_mode}")
        if args.max_examples:
            print(f"📊 Max examples: {args.max_examples}")
    
    # Update sys.path for reward_score module
    if args.reward_score_path not in sys.path:
        sys.path.insert(0, args.reward_score_path)
    
    # Load ground truth data and model responses
    ground_truth_data = load_jsonl_file(args.ground_truth_file, "ground truth")
    response_data = load_jsonl_file(args.input_file, "response")
    
    if not ground_truth_data:
        print("❌ No ground truth data loaded")
        sys.exit(1)
    
    if not response_data:
        print("❌ No response data loaded")
        sys.exit(1)
    
    # Create prompt-to-response mapping
    prompt_to_response = create_prompt_to_response_dict(response_data)
    
    # Match ground truth with responses
    matched_data = match_ground_truth_with_responses(ground_truth_data, prompt_to_response)
    
    if not matched_data:
        print("❌ No matching data found between ground truth and responses")
        sys.exit(1)
    
    # Limit examples if specified
    if args.max_examples and args.max_examples < len(matched_data):
        matched_data = matched_data[:args.max_examples]
        if not args.quiet:
            print(f"🔢 Limited to first {args.max_examples} examples")
    
    if not args.quiet:
        print(f"\n🔄 Evaluating {len(matched_data)} responses...")
    
    # Evaluate all responses sequentially
    results = []
    failed_count = 0
    
    # Process each response with progress bar
    desc = "Evaluating responses" if not args.quiet else None
    for response in tqdm(matched_data, desc=desc, disable=args.quiet):
        result = evaluate_single_response(response, args.framework, strict=not args.loose)
        results.append(result)
    
    if not args.quiet:
        print(f"\n✅ Evaluation completed!")
        print(f"   Processed: {len(results)} responses")
        print(f"   Evaluation errors: {failed_count}")
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print results (unless quiet mode)
    if not args.quiet:
        print_results(stats, model_name)
    
    # Save detailed results
    save_detailed_results(results, stats, args.output_file, model_name, evaluation_mode, args.framework)
    
    # Print summary
    print(f"\n🎉 Evaluation complete!")
    prompt_acc = stats['prompt_level_accuracy'] * 100
    inst_acc = stats['instruction_level_accuracy'] * 100
    framework_upper = args.framework.upper()
    print(f"📈 {model_name.upper()} achieved {prompt_acc:.2f}% prompt-level accuracy on {framework_upper} ({evaluation_mode} mode)")
    print(f"📊 Instruction-level accuracy: {inst_acc:.2f}%")
    
    if args.quiet:
        # In quiet mode, print just the key metrics
        prompt_acc_val = stats['prompt_level_accuracy']
        inst_acc_val = stats['instruction_level_accuracy']
        print(f"Results: {prompt_acc_val:.4f} prompt accuracy, {inst_acc_val:.4f} instruction accuracy")


if __name__ == "__main__":
    main() 