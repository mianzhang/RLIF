import os
from huggingface_hub import snapshot_download


model_cache_dir = 'hf_cache'

model_pool = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B'
    ]

for repo_id in model_pool:
    snapshot_download(
        repo_id,
        local_dir=os.path.join(model_cache_dir, repo_id),
    )
