# RLIF Evaluation Scripts

This directory contains evaluation scripts for assessing model performance on instruction following and algorithmic reasoning benchmarks. The scripts provide comprehensive evaluation capabilities with parallel processing, detailed metrics, and flexible configuration options.

## 📋 Overview

The evaluation framework supports three main benchmarks:

1. **LogicIF Mini** - Algorithmic reasoning and manual function execution
2. **IFBench** - Instruction following evaluation with comprehensive instruction types
3. **IFEval** - Instruction following evaluation with focus on verifiable constraints

## 🚀 Quick Start

### Prerequisites

- Python environment with required dependencies (tqdm, concurrent.futures)
- OpenAI API key (for LogicIF Mini extraction)
- VERL reward scoring modules properly installed

### Basic Usage

```bash
# LogicIF Mini evaluation
python evaluate_logicif.py --input_file model_responses.jsonl

# IFBench evaluation (strict mode)
python evaluate_if.py --framework ifbench --input_file responses.jsonl

# IFEval evaluation (loose mode)  
python evaluate_if.py --framework ifeval --input_file responses.jsonl --loose
```

## 📁 Evaluation Scripts

### 1. LogicIF Mini (`evaluate_logicif.py`)

Evaluates algorithmic reasoning capabilities by testing manual function execution.

**Key Features:**
- OpenAI-powered structured data extraction (default: gpt-5-nano)
- Parallel API calls for faster processing
- Both output and execution statistics validation
- Task-level and instance-level accuracy metrics

**Usage:**
```bash
# Basic evaluation
python evaluate_logicif.py

# Custom configuration
python evaluate_logicif.py \
  --input_file custom_responses.jsonl \
  --extract_model gpt-4o-mini \
  --max_workers 30 \
  --config_file config.json

# Quick test
python evaluate_logicif.py --max_examples 10 --quiet
```

**Parameters:**
- `--input_file, -i`: Input JSONL file with model responses
- `--output_file, -o`: Output JSON file for detailed results
- `--config_file, -c`: Configuration file with OpenAI API key
- `--extract_model, -e`: OpenAI model for extraction (default: gpt-5-nano)
- `--max_workers, -w`: Parallel workers (default: 40)
- `--max_examples, -n`: Limit number of examples
- `--model_name, -m`: Name of evaluated model
- `--quiet, -q`: Suppress detailed output

### 2. Instruction Following (`evaluate_if.py`)

Unified evaluation for both IFBench and IFEval instruction following benchmarks.

**Key Features:**
- Support for both IFBench and IFEval frameworks
- Strict and loose evaluation modes
- Parallel processing for faster evaluation
- Comprehensive instruction-level analysis

**Usage:**
```bash
# IFBench evaluation
python evaluate_if.py --framework ifbench

# IFEval with custom settings
python evaluate_if.py \
  --framework ifeval \
  --input_file responses.jsonl \
  --loose \
  --max_workers 50

# Quiet mode for batch processing
python evaluate_if.py --framework ifbench --max_examples 100 --quiet
```

**Parameters:**
- `--framework, -f`: **Required** - Choose 'ifbench' or 'ifeval'
- `--input_file, -i`: Input JSONL file with model responses
- `--output_file, -o`: Output JSON file for detailed results
- `--max_workers, -w`: Parallel workers (default: 40)
- `--max_examples, -n`: Limit number of examples
- `--model_name, -m`: Name of evaluated model
- `--loose`: Use loose evaluation mode (default: strict)
- `--quiet, -q`: Suppress detailed output

## 📊 Input Data Format

### LogicIF Mini Format
```json
{
  "task_id": "CODEFORCES_883A",
  "test_case_id": 0,
  "llm_response": "Step-by-step execution...",
  "code_output": {
    "output": 20,
    "stats": {"used_rope": true, "rope_units": 5}
  }
}
```

### IFBench/IFEval Format
```json
{
  "prompt": "Write a story with exactly 5 sentences.",
  "response": "Model's response text...",
  "instruction_ids": ["length_constraints:number_sentences"],
  "kwargs": [{"num_sentences": 5}],
  "model_used": "model_name"
}
```

## 📈 Output Metrics

### LogicIF Mini Metrics
- **Overall Accuracy**: Percentage where both output and stats match
- **Output Accuracy**: Percentage with correct final results
- **Stats Accuracy**: Percentage with correct execution statistics
- **Task-Level Accuracy**: Percentage of tasks where all test cases pass

### Instruction Following Metrics
- **Prompt-Level Accuracy**: Percentage of prompts with all instructions followed
- **Instruction-Level Accuracy**: Percentage of individual instructions followed
- **Performance by Instruction Count**: Breakdown by number of instructions per prompt

## ⚙️ Configuration

### OpenAI API Setup (LogicIF Mini)

Create `config.json`:
```json
{
  "OPENAI_API_KEY": "your-api-key-here"
}
```

### Default File Locations
- **LogicIF Mini**: `./RLIF_data/LogicIF/model_res/gpt_5-logic-if-eval-mini.jsonl`
- **IFBench**: `./RLIF_data/IFBench/data/sample_output.jsonl`
- **IFEval**: `./RLIF_data/IFEval/data/input_response_data_gpt4_20231107_145030.jsonl`

## 🔧 Advanced Usage

### Parallel Processing Optimization
```bash
# High-throughput evaluation
python evaluate_logicif.py --max_workers 50 --extract_model gpt-4o-mini

# Memory-conscious evaluation
python evaluate_if.py --framework ifbench --max_workers 10 --max_examples 100
```

### Batch Processing
```bash
# Process multiple files
for framework in ifbench ifeval; do
  python evaluate_if.py --framework $framework --quiet --model_name "batch-eval"
done

# Subset testing for quick validation
python evaluate_logicif.py --max_examples 50 --quiet
```

### Custom Output Locations
```bash
# Organized results directory
mkdir -p results/model_v1/
python evaluate_if.py \
  --framework ifbench \
  --output_file results/model_v1/ifbench_results.json
```

## 📋 Output Structure

### Detailed Results JSON
```json
{
  "evaluation_summary": {
    "total_examples": 100,
    "average_score": 0.85,
    "prompt_level_accuracy": 0.80,
    "instruction_level_accuracy": 0.90
  },
  "individual_results": [...],
  "evaluation_settings": {
    "model_evaluated": "gpt-4",
    "evaluation_framework": "ifbench",
    "evaluation_mode": "strict"
  }
}
```

## 🛠️ Integration with VERL

The evaluation scripts integrate with VERL's reward scoring framework:

```python
from verl.utils.reward_score import default_compute_score

# LogicIF Mini
score = default_compute_score(
    solution_str="model response",
    ground_truth={"task_id": "...", "code_output": {...}},
    data_source="logicifeval-mini"
)

# IFBench
score = default_compute_score(
    solution_str="model response", 
    ground_truth={"instruction_ids": [...], "kwargs": [...]},
    data_source="ifbench"
)
```

## 🐛 Troubleshooting

### Common Issues

**OpenAI API Errors (LogicIF Mini)**
```bash
# Check API key configuration
cat config.json

# Test with fewer workers to avoid rate limits
python evaluate_logicif.py --max_workers 5
```

**Memory Issues**
```bash
# Process in smaller batches
python evaluate_if.py --framework ifbench --max_examples 50
```

**Module Import Errors**
```bash
# Verify reward_score path
python evaluate_if.py --reward_score_path ./verl/utils/reward_score
```

### Performance Optimization

1. **Adjust worker count** based on system capabilities and API rate limits
2. **Use appropriate extraction models** - gpt-4o-mini for speed, gpt-5-nano for accuracy
3. **Enable quiet mode** for batch processing to reduce I/O overhead
4. **Process subsets** for testing before full evaluation runs

## 📚 Examples

### Complete Evaluation Workflow
```bash
# 1. Prepare data and configuration
mkdir -p results/
cp model_responses.jsonl ./

# 2. Run evaluations
python evaluate_logicif.py \
  --input_file model_responses.jsonl \
  --output_file results/logicif_results.json \
  --model_name "MyModel-v1"

python evaluate_if.py \
  --framework ifbench \
  --input_file model_responses.jsonl \
  --output_file results/ifbench_results.json \
  --model_name "MyModel-v1"

python evaluate_if.py \
  --framework ifeval \
  --input_file model_responses.jsonl \
  --output_file results/ifeval_results.json \
  --loose \
  --model_name "MyModel-v1"

# 3. Compare results
python -c "
import json
for f in ['logicif_results.json', 'ifbench_results.json', 'ifeval_results.json']:
    with open(f'results/{f}') as file:
        data = json.load(file)
        summary = data['evaluation_summary']
        print(f'{f}: {summary.get(\"average_score\", summary.get(\"prompt_level_accuracy\", 0)):.3f}')
"
```

## 🔍 Quick Reference

### Current Evaluation Scripts
| Script | Framework | Purpose | Key Features |
|--------|-----------|---------|-------------|
| `evaluate_logicif.py` | LogicIF Mini | Algorithmic reasoning | OpenAI extraction, parallel processing |
| `evaluate_if.py` | IFBench/IFEval | Instruction following | Unified interface, strict/loose modes |

### Common Parameters
| Parameter | LogicIF | IF Scripts | Description |
|-----------|---------|------------|-------------|
| `--input_file, -i` | ✅ | ✅ | Input JSONL file |
| `--output_file, -o` | ✅ | ✅ | Output JSON file |
| `--max_workers, -w` | ✅ | ✅ | Parallel workers |
| `--max_examples, -n` | ✅ | ✅ | Limit examples |
| `--model_name, -m` | ✅ | ✅ | Model name |
| `--quiet, -q` | ✅ | ✅ | Suppress output |
| `--framework, -f` | ❌ | ✅ | Required framework choice |
| `--extract_model, -e` | ✅ | ❌ | OpenAI extraction model |
| `--config_file, -c` | ✅ | ❌ | OpenAI API key config |
| `--loose` | ❌ | ✅ | Loose evaluation mode |

### Typical Workflows

**Quick Testing:**
```bash
python evaluate_logicif.py --max_examples 10 --quiet
python evaluate_if.py --framework ifbench --max_examples 10 --quiet
```

**Production Evaluation:**
```bash
python evaluate_logicif.py --model_name "Production-Model-v1"
python evaluate_if.py --framework ifbench --model_name "Production-Model-v1"
python evaluate_if.py --framework ifeval --loose --model_name "Production-Model-v1"
```

**High-Throughput Processing:**
```bash
python evaluate_logicif.py --max_workers 50 --extract_model gpt-4o-mini
python evaluate_if.py --framework ifbench --max_workers 50
```

## 📄 License

This evaluation framework is part of the RLIF project and follows the same licensing terms.

---

For more detailed information about the underlying evaluation frameworks, see:
- `verl/utils/reward_score/IF_REWARD_FUNC.md` - Comprehensive reward function documentation
- Individual benchmark documentation in `RLIF_data/` directories 