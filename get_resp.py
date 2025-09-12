import llminfer

from constant import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


for model in [QWEN3_8B]:
    for benchmark in [IFEVAL, IFBENCH, LOGICIFEVALMINI]:
        llminfer.process_jsonl(
            benchmark.prompt_path,
            f'eval_res/{benchmark.nickname}-{model.nickname}.jsonl',
            provider="vllm",
            model=model.model_path,
            input_key=benchmark.input_key,  # Key pointing to string prompts
            temperature=0.7,
        )
