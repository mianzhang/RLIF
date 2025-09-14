#!/usr/bin/env python3
"""
Quick script to check token lengths for parquet files.
Simplified version for fast analysis.

Usage:
    python quick_token_check.py file.parquet
    python quick_token_check.py file.parquet --tokenizer /path/to/tokenizer
"""

import sys
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_token_check.py <parquet_file> [--tokenizer <tokenizer_path>]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    tokenizer_path = '/localdisk/models/Qwen/Qwen3-0.6B'
    
    # Parse tokenizer argument
    if '--tokenizer' in sys.argv:
        idx = sys.argv.index('--tokenizer')
        if idx + 1 < len(sys.argv):
            tokenizer_path = sys.argv[idx + 1]
    
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
        over_512 = sum(1 for x in token_lengths if x > 512)
        over_1024 = sum(1 for x in token_lengths if x > 1024)
        
        print(f"\n{data_source} ({len(subset)} samples):")
        print(f"  Range: {min_len} - {max_len} tokens (avg: {mean_len:.0f})")
        print(f"  95th percentile: {p95_len:.0f}")
        print(f"  > 512 tokens: {over_512}/{len(subset)} ({over_512/len(subset)*100:.1f}%)")
        print(f"  > 1024 tokens: {over_1024}/{len(subset)} ({over_1024/len(subset)*100:.1f}%)")
    
    # Overall recommendation
    all_lengths = []
    for prompt in df['prompt']:
        tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        all_lengths.append(len(tokens))
    
    overall_max = max(all_lengths)
    recommended = int(overall_max * 1.1)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Set data.max_prompt_length={recommended} (current max: {overall_max})")

if __name__ == '__main__':
    main() 