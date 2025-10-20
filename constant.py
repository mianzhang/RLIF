from dataclasses import dataclass


HF_CACHE_DIR = "hf_cache"

# Models
@dataclass
class QWEN3_06B:
    nickname = "qwen06b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-0.6B"

@dataclass
class QWEN3_17B:
    nickname = "qwen17b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-1.7B"

@dataclass
class QWEN3_4B:
    nickname = "qwen4b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-4B"


@dataclass
class QWEN3_8B:
    nickname = "qwen8b"
    model_path = f"{HF_CACHE_DIR}/Qwen/Qwen3-8B"


# Training Data Mixture (for verl)
@dataclass # small data for testing
class MIX_IF1000_LOGICIF1000:
    nickname = "if1000_logicif1000"
    data_file = "verl_data/if1000_logicif1000.parquet"

@dataclass
class MIX_IF80000:
    nickname = "if80000"
    data_file = "verl_data/if80000.parquet"

@dataclass
class MIX_IF20000_LOGICIF60000:
    nickname = "if20000_logicif60000"
    data_file = "verl_data/if20000_logicif60000.parquet"

@dataclass
class MIX_IF40000_LOGICIF40000:
    nickname = "if40000_logicif40000"
    data_file = "verl_data/if40000_logicif40000.parquet"

@dataclass
class MIX_IF60000_LOGICIF20000:
    nickname = "if60000_logicif20000"
    data_file = "verl_data/if60000_logicif20000.parquet"

@dataclass
class MIX_LOGICIF80000:
    nickname = "logicif80000"
    data_file = "verl_data/logicif80000.parquet"


# Eval Data Mixture (for verl)
@dataclass
class RULEIFEVAL:
    nickname = "ruleifeval"
    data_file = "verl_data/ifbench_294_ifeval_541_logicif_749.parquet"


# Evaluation Benchmarks
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