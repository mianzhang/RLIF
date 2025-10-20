import os
import argparse
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub")
parser.add_argument("folder_path", type=str, help="Path to the folder to upload")
parser.add_argument("--repo-id", type=str, default="billmianz/RLIF", help="Hugging Face repository ID (default: billmianz/RLIF)")
args = parser.parse_args()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=args.folder_path,
    repo_id=args.repo_id,
    repo_type="dataset",
)