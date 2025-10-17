# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Quick script to check token lengths for parquet files.
Simplified version for fast analysis.

Usage:
    python quick_token_check.py file.parquet
    python quick_token_check.py file.parquet --tokenizer /path/to/tokenizer
    python quick_token_check.py file.parquet --filter 1024 --output filtered.parquet
"""

import argparse
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Check token lengths for parquet files and optionally filter by token count.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python quick_token_check.py data.parquet

  # With custom tokenizer
  python quick_token_check.py data.parquet --tokenizer /path/to/model

  # Filter rows with prompts ≤ 1024 tokens
  python quick_token_check.py data.parquet --filter 1024

  # Filter with custom output filename
  python quick_token_check.py data.parquet --filter 2048 --output short_prompts.parquet
        """
    )
    
    parser.add_argument(
        'file',
        type=str,
        help='Path to the parquet file to analyze'
    )
    
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='/localdisk/models/Qwen/Qwen3-0.6B',
        help='Path to the tokenizer (default: /localdisk/models/Qwen/Qwen3-0.6B)'
    )
    
    parser.add_argument(
        '--filter',
        type=int,
        metavar='MAX_TOKENS',
        help='Filter and save only rows with prompts ≤ MAX_TOKENS tokens'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        metavar='OUTPUT_FILE',
        help='Output filename for filtered data (auto-generated if not provided)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    file_path = args.file
    tokenizer_path = args.tokenizer
    filter_threshold = args.filter
    output_file = args.output
    
    print(f"📁 File: {file_path}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    
    # Load data and tokenizer
    df = pd.read_parquet(file_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print(f"📊 Total rows: {len(df)}")
    print(f"📋 Data sources: {df['data_source'].value_counts().to_dict()}")
    
    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS")
    print("="*60)
    
    for data_source in df['data_source'].unique():
        subset = df[df['data_source'] == data_source]
        
        # Calculate token lengths
        token_lengths = []
        for prompt in subset['prompt']:
            tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            token_lengths.append(len(tokens))
        
        # Quick stats
        min_len = min(token_lengths)
        max_len = max(token_lengths)
        mean_len = np.mean(token_lengths)
        p95_len = np.percentile(token_lengths, 95)
        
        # Count samples under different thresholds
        thresholds = [512, 1024, 2048, 4096, 8192]
        under_counts = {t: sum(1 for x in token_lengths if x <= t) for t in thresholds}
        
        print(f"\n{data_source} ({len(subset)} samples):")
        print(f"  Range: {min_len} - {max_len} tokens (avg: {mean_len:.0f})")
        print(f"  95th percentile: {p95_len:.0f}")
        for threshold in thresholds:
            count = under_counts[threshold]
            pct = count/len(subset)*100
            print(f"  ≤ {threshold} tokens: {count}/{len(subset)} ({pct:.1f}%)")
    
    # Overall recommendation
    all_lengths = []
    for prompt in df['prompt']:
        tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        all_lengths.append(len(tokens))
    
    overall_max = max(all_lengths)
    recommended = int(overall_max * 1.1)
    
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    thresholds = [512, 1024, 2048, 4096, 8192]
    for threshold in thresholds:
        count = sum(1 for x in all_lengths if x <= threshold)
        pct = count/len(all_lengths)*100
        print(f"  ≤ {threshold} tokens: {count}/{len(all_lengths)} ({pct:.1f}%)")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Set data.max_prompt_length={recommended} (current max: {overall_max})")
    
    # Filter and save if requested
    if filter_threshold is not None:
        print(f"\n{'='*60}")
        print(f"FILTERING ROWS WITH ≤ {filter_threshold} TOKENS")
        print(f"{'='*60}")
        
        # Create a mask for rows that pass the filter
        filtered_indices = [i for i, length in enumerate(all_lengths) if length <= filter_threshold]
        filtered_df = df.iloc[filtered_indices].reset_index(drop=True)
        
        # Auto-generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(file_path)[0]
            output_file = f"{base_name}_filtered_{filter_threshold}.parquet"
        
        # Save filtered data
        filtered_df.to_parquet(output_file, index=False)
        
        print(f"  Filtered: {len(filtered_df)}/{len(df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
        print(f"  Saved to: {output_file}")
        
        # Show breakdown by data source
        print(f"\n  Breakdown by data source:")
        for data_source in df['data_source'].unique():
            original_count = len(df[df['data_source'] == data_source])
            filtered_count = len(filtered_df[filtered_df['data_source'] == data_source])
            pct = filtered_count/original_count*100 if original_count > 0 else 0
            print(f"    {data_source}: {filtered_count}/{original_count} ({pct:.1f}%)")

if __name__ == '__main__':
    main() 