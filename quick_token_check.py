# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Quick script to check token lengths for parquet and jsonl files.
Simplified version for fast analysis.

Usage:
    python quick_token_check.py file.parquet
    python quick_token_check.py file.jsonl
    python quick_token_check.py file.parquet --tokenizer /path/to/tokenizer
    python quick_token_check.py file.parquet --save-lengths lengths.pkl
    python quick_token_check.py file.jsonl --column messages --save-lengths lengths.pkl
"""

import argparse
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import os
import json
import pickle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Check token lengths for parquet/jsonl files and optionally save length information.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (parquet)
  python quick_token_check.py data.parquet

  # Analyze JSONL file
  python quick_token_check.py data.jsonl --column messages

  # With custom tokenizer
  python quick_token_check.py data.parquet --tokenizer /path/to/model

  # Save length information to pickle file
  python quick_token_check.py data.parquet --save-lengths lengths.pkl

  # Save lengths for specific column
  python quick_token_check.py data.jsonl --column messages --save-lengths lengths.pkl
        """
    )
    
    parser.add_argument(
        'file',
        type=str,
        help='Path to the parquet or jsonl file to analyze'
    )
    
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='/localdisk/models/Qwen/Qwen3-0.6B',
        help='Path to the tokenizer (default: /localdisk/models/Qwen/Qwen3-0.6B)'
    )
    
    parser.add_argument(
        '--column',
        type=str,
        default='prompt',
        help='Column name to analyze for token length (default: prompt)'
    )
    
    parser.add_argument(
        '--save-lengths',
        type=str,
        metavar='OUTPUT_FILE',
        help='Save a dictionary mapping content to token length as pickle file'
    )
    
    parser.add_argument(
        '--force-chat-template',
        action='store_true',
        help='When saving lengths, convert plain text to conversation format first. '
             'This ensures keys match VERL format: [{"role": "user", "content": text}]'
    )
    
    return parser.parse_args()

def load_data(file_path):
    """Load data from parquet or jsonl file, auto-detecting the format."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.parquet':
        print(f"📂 Detected format: Parquet")
        return pd.read_parquet(file_path), 'parquet'
    elif file_ext in ['.jsonl', '.json']:
        print(f"📂 Detected format: JSONL")
        # Read jsonl file
        data = []
        # First count lines for progress bar
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading JSONL", unit="line"):
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return pd.DataFrame(data), 'jsonl'
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .parquet, .jsonl, .json")

def is_conversation_format(content):
    """
    Detect if content is in conversation format.
    Conversation format: list of dicts with 'role' and 'content' keys.
    """
    if isinstance(content, np.ndarray):
        content = content.tolist()
    
    if not isinstance(content, list):
        return False
    
    if len(content) == 0:
        return False
    
    # Check if it's a list of message dicts
    if isinstance(content[0], dict):
        # Should have 'role' and 'content' keys (or similar chat format)
        return 'role' in content[0] or 'content' in content[0]
    
    return False

def convert_to_conversation_format(content):
    """
    Convert plain text content to conversation format.
    Returns the conversation format: [{"role": "user", "content": text}]
    """
    if is_conversation_format(content):
        return content
    else:
        # Convert plain text to conversation format
        text = content if isinstance(content, str) else str(content)
        return [{"role": "user", "content": text}]

def tokenize_content(content, tokenizer, force_chat_template=False):
    """
    Tokenize content based on its format.
    - If conversation format: use apply_chat_template
    - If plain string: tokenize directly (or use chat template if forced)
    
    Args:
        content: Content to tokenize
        tokenizer: Tokenizer to use
        force_chat_template: If True, convert plain text to conversation format first
    """
    if force_chat_template and not is_conversation_format(content):
        # Convert to conversation format first
        content = convert_to_conversation_format(content)
    
    if is_conversation_format(content):
        # Use chat template for conversation format
        tokens = tokenizer.apply_chat_template(content, add_generation_prompt=True)
    else:
        # Direct tokenization for plain string
        if isinstance(content, str):
            tokens = tokenizer.encode(content, add_special_tokens=True)
        else:
            # Convert to string if needed
            tokens = tokenizer.encode(str(content), add_special_tokens=True)
    
    return tokens

def main():
    args = parse_args()
    
    file_path = args.file
    tokenizer_path = args.tokenizer
    save_lengths_file = args.save_lengths
    column_name = args.column
    
    print(f"📁 File: {file_path}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    print(f"📋 Column to analyze: {column_name}")
    
    # Load data and tokenizer
    df, file_format = load_data(file_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Verify column exists
    if column_name not in df.columns:
        print(f"\n✗ Error: Column '{column_name}' not found in file")
        print(f"Available columns: {', '.join(df.columns)}")
        return 1
    
    print(f"📊 Total rows: {len(df)}")
    
    # Check if data_source column exists
    has_data_source = 'data_source' in df.columns
    if has_data_source:
        print(f"📋 Data sources: {df['data_source'].value_counts().to_dict()}")
    
    # Detect content format from first non-null entry
    first_content = df[column_name].dropna().iloc[0] if len(df[column_name].dropna()) > 0 else None
    is_conversation = is_conversation_format(first_content) if first_content is not None else False
    format_type = "conversation (chat template)" if is_conversation else "plain text"
    print(f"🔍 Detected content format: {format_type}")
    
    # Check if force_chat_template is enabled
    if args.force_chat_template and not is_conversation:
        print(f"⚠️  --force-chat-template enabled: will convert plain text to conversation format")
        format_type = "plain text → conversation (chat template)"
    
    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS")
    print("="*60)
    
    # Calculate token lengths once for all content
    print("Tokenizing content...")
    all_lengths = []
    for content in tqdm(df[column_name], desc="Tokenizing", unit="item"):
        tokens = tokenize_content(content, tokenizer, force_chat_template=args.force_chat_template)
        all_lengths.append(len(tokens))
    print(f"✓ Tokenized {len(all_lengths)} items")
    
    # Analyze by data_source if available, otherwise analyze all data together
    if has_data_source:
        data_groups = [(source, df[df['data_source'] == source]) for source in df['data_source'].unique()]
    else:
        data_groups = [("all_data", df)]
    
    for group_name, subset in data_groups:
        # Get token lengths for this subset using indices
        subset_indices = subset.index.tolist()
        token_lengths = [all_lengths[i] for i in subset_indices]
        
        # Quick stats
        min_len = min(token_lengths)
        max_len = max(token_lengths)
        mean_len = np.mean(token_lengths)
        p95_len = np.percentile(token_lengths, 95)
        
        # Count samples under different thresholds
        thresholds = [512, 1024, 2048, 4096, 8192]
        under_counts = {t: sum(1 for x in token_lengths if x <= t) for t in thresholds}
        
        print(f"\n{group_name} ({len(subset)} samples):")
        print(f"  Range: {min_len} - {max_len} tokens (avg: {mean_len:.0f})")
        print(f"  95th percentile: {p95_len:.0f}")
        for threshold in thresholds:
            count = under_counts[threshold]
            pct = count/len(subset)*100
            print(f"  ≤ {threshold} tokens: {count}/{len(subset)} ({pct:.1f}%)")
    
    overall_max = max(all_lengths)
    
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    thresholds = [512, 1024, 2048, 4096, 8192]
    for threshold in thresholds:
        count = sum(1 for x in all_lengths if x <= threshold)
        pct = count/len(all_lengths)*100
        print(f"  ≤ {threshold} tokens: {count}/{len(all_lengths)} ({pct:.1f}%)")
    
    print(f"  Set data.max_prompt_length={overall_max}) + N")
    
    # Save length information if requested
    if save_lengths_file is not None:
        print(f"\n{'='*60}")
        print(f"SAVING LENGTH INFORMATION")
        print(f"{'='*60}")
        
        # Create a dictionary mapping content to token length
        # For lists (like chat messages), we'll use the original content as key
        # Pickle can handle complex Python objects unlike JSON
        print("Creating length dictionary...")
        length_dict = {}
        for i, content in enumerate(tqdm(df[column_name], desc="Building dict", unit="item")):
            # If force_chat_template is enabled, convert to conversation format first
            if args.force_chat_template and not is_conversation_format(content):
                # Convert plain text to conversation format for the key
                content_for_key = convert_to_conversation_format(content)
            else:
                content_for_key = content
            
            # Convert content to a hashable representation for use as dict key
            if isinstance(content_for_key, list):
                # For list content (e.g., messages), use JSON string as key
                key = json.dumps(content_for_key, ensure_ascii=False, sort_keys=True)
            else:
                # For string content, use as is
                key = str(content_for_key)
            
            # Use already computed token length
            length_dict[key] = all_lengths[i]
        
        # Save to pickle file
        print("Saving to pickle file...")
        with open(save_lengths_file, 'wb') as f:
            pickle.dump(length_dict, f)
        
        print(f"  ✓ Saved {len(length_dict)} content-to-length mappings")
        print(f"  ✓ Output file: {save_lengths_file}")
        if args.force_chat_template:
            print(f"  ℹ️  Keys saved in conversation format: [{{\"role\": \"user\", \"content\": ...}}]")
        
        # Show some statistics
        unique_contents = len(length_dict)
        total_rows = len(df)
        if unique_contents < total_rows:
            print(f"  ℹ️  Note: {total_rows} rows, {unique_contents} unique contents")
    
    return 0

if __name__ == '__main__':
    exit(main()) 