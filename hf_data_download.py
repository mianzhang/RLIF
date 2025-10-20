#!/usr/bin/env python3
"""
Script to download all data files from the RLIF Hugging Face repository.
Downloads all files from billmianz/RLIF into verl_data/ directory.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    repo_id = "billmianz/RLIF"
    local_dir = "verl_data"
    
    hf_token = os.environ.get("HF_TOKEN")
    
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
    )
    
if __name__ == "__main__":
    exit(main()) 