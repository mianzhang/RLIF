# verl_extensions: Multi-Source Curriculum Samplers

Custom extensions for the verl framework enabling multi-source data sampling with curriculum learning.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Samplers](#samplers)
- [Configuration Reference](#configuration-reference)
- [Usage Examples](#usage-examples)
- [Monitoring & Debugging](#monitoring--debugging)
- [Data Requirements](#data-requirements)

---

## Overview

This package provides three samplers for training on multiple data sources:

| Sampler | Description | Use Case |
|---------|-------------|----------|
| **MultiSourceSampler** | Fixed or adaptive source weights | Known ratios or automatic balancing |
| **TimeCurriculumSampler** | Time-based complexity progression | Scheduled difficulty increase |
| **RewardCurriculumSampler** | Reward-gated complexity progression | Performance-based difficulty |

### Key Features

✅ Mix multiple data sources with configurable ratios  
✅ Fixed or adaptive source weights (lower reward → higher weight)  
✅ Time-based complexity curriculum (linear progression)  
✅ Reward-gated complexity curriculum (unlock when ready)  
✅ Seamless verl integration via configuration  

---

## Quick Start

### 1. Fixed Weights (1:1 Ratio)

```bash
python3 -m verl.trainer.main_ppo \
    data.sampler.class_name=MultiSourceSampler \
    +data.sampler.sampling_strategy=weighted \
    +data.sampler.source_weights.source1=1.0 \
    +data.sampler.source_weights.source2=1.0 \
    ...
```

### 2. Adaptive Weights (Auto-Balance)

```bash
python3 -m verl.trainer.main_ppo \
    data.sampler.class_name=MultiSourceSampler \
    +data.sampler.adaptive_weights=true \
    +data.sampler.alpha=0.1 \
    +data.sampler.temperature=0.3 \
    ...
```

### 3. Time-Based Curriculum

```bash
python3 -m verl.trainer.main_ppo \
    data.sampler.class_name=TimeCurriculumSampler \
    +data.sampler.complexity_source=logicif \
    +data.sampler.initial_complexity_percentile=30 \
    +data.sampler.final_complexity_percentile=100 \
    +data.sampler.complexity_warmup_steps=100 \
    ...
```

### 4. Reward-Gated Curriculum

```bash
python3 -m verl.trainer.main_ppo \
    data.sampler.class_name=RewardCurriculumSampler \
    +data.sampler.complexity_source=logicif \
    +data.sampler.initial_complexity_percentile=30 \
    +data.sampler.num_intervals=10 \
    +data.sampler.reward_threshold=0.125 \
    ...
```

---

## Samplers

### MultiSourceSampler

**Purpose**: Mix multiple data sources with fixed or adaptive weights.

**Features**:
- Fixed weights: Constant sampling ratio (e.g., 1:1, 2:1)
- Adaptive weights: Automatically focus on harder sources (lower reward → higher weight)

**Configuration**:
```yaml
sampler:
  class_name: MultiSourceSampler
  sampling_strategy: weighted
  source_weights:
    source1: 1.0
    source2: 1.0
  
  # Optional: Enable adaptive weights
  adaptive_weights: true
  alpha: 0.1                # EMA smoothing (0.05-0.2)
  update_frequency: 5       # Update every N batches
  temperature: 0.3          # Lower = more sensitive
  max_source_ratio: 3.0     # Max weight ratio
```

---

### TimeCurriculumSampler

**Purpose**: Start with easy data, linearly increase difficulty over time.

**How it works**:
1. Start with `initial_complexity_percentile` of easiest data
2. Linearly increase to `final_complexity_percentile` over `complexity_warmup_steps`
3. Optionally use adaptive source weights

**Configuration**:
```yaml
sampler:
  class_name: TimeCurriculumSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
  
  # Optional: Adaptive weights
  adaptive_weights: true
  alpha: 0.1
  temperature: 0.3
  max_source_ratio: 3.0
  
  # Curriculum config
  complexity_source: logicif
  initial_complexity_percentile: 30
  final_complexity_percentile: 100
  complexity_warmup_steps: 100
```

---

### RewardCurriculumSampler

**Purpose**: Unlock harder data only when model achieves performance threshold.

**How it works**:
1. Start with `initial_complexity_percentile` of easiest data
2. Divide remaining data into `num_intervals` levels
3. Unlock next level when reward EMA exceeds `reward_threshold`
4. Optionally use adaptive source weights

**Configuration**:
```yaml
sampler:
  class_name: RewardCurriculumSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
  
  # Optional: Adaptive weights
  adaptive_weights: false
  
  # Curriculum config
  complexity_source: logicif
  initial_complexity_percentile: 30
  final_complexity_percentile: 100
  num_intervals: 10
  reward_threshold: 0.125
  reward_alpha: 0.1
  min_batches_per_level: 10
  check_frequency: 3
```

---

## Configuration Reference

### Common Parameters (All Samplers)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_path` | str | required | `pkg://verl_extensions.multi_source_sampler` |
| `class_name` | str | required | `MultiSourceSampler`, `TimeCurriculumSampler`, or `RewardCurriculumSampler` |
| `sampling_strategy` | str | `weighted` | `weighted` or `proportional` |
| `source_weights` | dict | `{}` | Weight per source |

### Adaptive Weights Parameters (All Samplers)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adaptive_weights` | bool | `false` | Enable adaptive source weights |
| `alpha` | float | `0.1` | EMA smoothing for reward tracking |
| `update_frequency` | int | `5` | Batches between weight updates |
| `temperature` | float | `1.0` | Softmax temperature (lower = more sensitive) |
| `max_source_ratio` | float | `3.0` | Max ratio between source weights |

### TimeCurriculumSampler Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexity_source` | str | `logicif` | Source to apply curriculum |
| `initial_complexity_percentile` | int | `30` | Start with easiest X% |
| `final_complexity_percentile` | int | `100` | End with X% |
| `complexity_warmup_steps` | int | `100` | Updates to reach final |

### RewardCurriculumSampler Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexity_source` | str | `logicif` | Source to apply curriculum |
| `initial_complexity_percentile` | int | `30` | Start with easiest X% |
| `final_complexity_percentile` | int | `100` | Maximum X% to use |
| `num_intervals` | int | `10` | Intervals between initial and final |
| `reward_threshold` | float | required | Threshold to unlock next level |
| `reward_alpha` | float | `0.1` | EMA smoothing for curriculum rewards |
| `min_batches_per_level` | int | `10` | Min batches before checking |
| `check_frequency` | int | `3` | Check every N batches |
| `focus_new_intervals` | bool | `false` | Give higher sampling weight to newer intervals |
| `interval_decay` | float | `0.5` | Decay factor for older intervals (0-1) |

---

## Usage Examples

### Example 1: Fixed 1:1 Ratio

```yaml
sampler:
  class_name: MultiSourceSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
```

### Example 2: Adaptive Source Weights

```yaml
sampler:
  class_name: MultiSourceSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
  adaptive_weights: true
  alpha: 0.1
  temperature: 0.3
  max_source_ratio: 3.0
```

### Example 3: Time-Based Curriculum with Adaptive Weights

```yaml
sampler:
  class_name: TimeCurriculumSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
  adaptive_weights: true
  alpha: 0.1
  complexity_source: logicif
  initial_complexity_percentile: 30
  final_complexity_percentile: 100
  complexity_warmup_steps: 100
```

### Example 4: Reward-Gated Curriculum

```yaml
sampler:
  class_name: RewardCurriculumSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
  complexity_source: logicif
  initial_complexity_percentile: 20
  final_complexity_percentile: 80
  num_intervals: 10
  reward_threshold: 0.125
  reward_alpha: 0.1
  min_batches_per_level: 20
  check_frequency: 5
```

### Example 5: Reward-Gated with Focus on New Intervals

```yaml
sampler:
  class_name: RewardCurriculumSampler
  sampling_strategy: weighted
  source_weights:
    iftrain: 1.0
    logicif: 1.0
  complexity_source: logicif
  initial_complexity_percentile: 20
  final_complexity_percentile: 100
  num_intervals: 10
  reward_threshold: 0.125
  # Focus more on newly unlocked intervals
  focus_new_intervals: true
  interval_decay: 0.5  # Older intervals have 50% less weight per level
```

With `interval_decay=0.5` and 3 intervals unlocked:
- Interval 2 (newest): weight = 1.0 → 57%
- Interval 1: weight = 0.5 → 29%
- Interval 0 (oldest): weight = 0.25 → 14%

---

## Monitoring & Debugging

### Log Output Examples

**MultiSourceSampler (Fixed)**:
```
[MultiSourceSampler] Strategy: weighted, Mode: fixed, Sources: 2, Batch size: 512
  iftrain: 40000 samples (weight: 0.500)
  logicif: 40000 samples (weight: 0.500)
```

**MultiSourceSampler (Adaptive)**:
```
[MultiSourceSampler] Strategy: weighted, Mode: adaptive, Sources: 2, Batch size: 512
[MultiSourceSampler] Adaptive: alpha=0.1, update_freq=5, temp=0.3
[MultiSourceSampler] Weight bounds: [25.0%, 75.0%]

[MultiSourceSampler] Batch 50 - Weight adjustment:
  iftrain: reward_ema= 10.25, weight=0.450 (45.0%)
  logicif: reward_ema=  8.73, weight=0.550 (55.0%)
  Actual max/min ratio: 1.22x
```

**TimeCurriculumSampler**:
```
[TimeCurriculumSampler] Complexity curriculum for 'logicif'
[TimeCurriculumSampler] Progress: 30% → 100% over 100 updates

[TimeCurriculumSampler] Complexity: 50% (40000/80000 items)
```

**RewardCurriculumSampler**:
```
[RewardCurriculumSampler] Complexity source: 'logicif'
[RewardCurriculumSampler] Initial: 30%, Final: 100%
[RewardCurriculumSampler] Intervals: 11 (1 initial + 10 progressive)
  Interval 0 (Initial 30%): 24000 samples, [0.123, 0.412]
  Interval 1: 5600 samples, [0.412, 0.489]
  ...

================================================================================
[RewardCurriculumSampler] LEVEL UP! 0 → 1
[RewardCurriculumSampler] Batch 82: Reward 0.128 >= 0.125
[RewardCurriculumSampler] Samples: 24000 → 29600
================================================================================
```

---

## Data Requirements

### Required Fields (All Samplers)

```python
{
    'prompt': 'Your prompt...',
    'response': 'Expected response...',
    'data_source': 'source_name'  # Must match source_weights keys
}
```

### Additional Fields (Curriculum Samplers)

```python
{
    'prompt': '...',
    'data_source': 'logicif',
    'extra_info': {
        'complexity_score': 0.73  # Float value for curriculum
    }
}
```

---

## Backwards Compatibility

Old sampler names still work:
- `CurriculumMultiSourceSampler` → `TimeCurriculumSampler`
- `RewardGatedCurriculumSampler` → `RewardCurriculumSampler`

---

## Files

- `multi_source_sampler.py` - All sampler implementations
- `__init__.py` - Package exports
- `README.md` - This file
- `example_config.yaml` - Configuration examples
- `SOURCE_TRACKING.md` - Per-source metrics tracking

---

## Version History

- **0.3.0** (Current): Refactored samplers, added adaptive_weights to base class
- **0.2.0**: Added RewardCurriculumSampler
- **0.1.0**: Initial release

---

## License

Apache 2.0 (same as verl framework)
