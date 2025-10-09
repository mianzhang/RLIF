from dataclasses import dataclass

HF_CACHE_DIR = "/localdisk/models"


@dataclass
class QWEN3_0_6B:
    nickname = "qwen306b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-0.6B"

@dataclass
class QWEN3_1_7B:
    nickname = "qwen317b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-1.7B"

@dataclass
class QWEN3_4B:
    nickname = "qwen34b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-4B"


@dataclass
class QWEN3_8B:
    nickname = "qwen38b"
    model_path = f"{HF_CACHE_DIR}/Qwen3-8B"



@dataclass
class IFBENCH:
    nickname = "ifbench"
    input_key = "prompt"
    prompt_path = "benchmark/ifbench.jsonl"

@dataclass
class IFEVAL:
    nickname = "ifeval"
    input_key = "prompt"
    prompt_path = "benchmark/ifeval.jsonl"

@dataclass
class LOGICIFEVALMINI:
    nickname = "logicifevalmini"
    input_key = 'instruction' 
    prompt_path = "benchmark/logicifevalmini.jsonl"

@dataclass
class INFORBENCH:
    nickname = "infobench"
    input_key = "instruction"
    prompt_path = "benchmark/infobench.jsonl"