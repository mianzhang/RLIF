# Changelog

## Version 0.3.0 - Sampler Refactoring

### Major Changes

#### Renamed Samplers
- `CurriculumMultiSourceSampler` → `TimeCurriculumSampler`
- `RewardGatedCurriculumSampler` → `RewardCurriculumSampler`
- Old names still work as aliases for backwards compatibility

#### MultiSourceSampler Enhancements
- Added `adaptive_weights` parameter to base MultiSourceSampler
- When `adaptive_weights=true`, source weights are dynamically adjusted based on rewards
- Sources with lower rewards get higher weights (focus on harder sources)
- This consolidates the weight adaptation logic that was previously only in CurriculumMultiSourceSampler

#### Cleaner Code Structure
- Extracted `_CurriculumSamplerMixin` for shared complexity extraction logic
- Both curriculum samplers now inherit from MultiSourceSampler
- Both curriculum samplers support adaptive_weights (optional)

### New Configuration

**MultiSourceSampler with Adaptive Weights**:
```yaml
sampler:
  class_name: MultiSourceSampler
  source_weights:
    source1: 1.0
    source2: 1.0
  adaptive_weights: true  # NEW: Enable dynamic weight adjustment
  alpha: 0.1
  update_frequency: 5
  temperature: 0.3
  max_source_ratio: 3.0
```

**TimeCurriculumSampler** (renamed from CurriculumMultiSourceSampler):
```yaml
sampler:
  class_name: TimeCurriculumSampler
  complexity_source: logicif
  initial_complexity_percentile: 30
  final_complexity_percentile: 100
  complexity_warmup_steps: 100
  # Optional: adaptive source weights
  adaptive_weights: true
```

**RewardCurriculumSampler** (renamed from RewardGatedCurriculumSampler):
```yaml
sampler:
  class_name: RewardCurriculumSampler
  complexity_source: logicif
  initial_complexity_percentile: 30
  num_intervals: 10
  reward_threshold: 0.125
```

### Breaking Changes
- Removed `use_complexity_curriculum` parameter (complexity curriculum always enabled for curriculum samplers)
- Removed redundant `uniform` sampling strategy (use `weighted` with equal weights)

---

## Version 0.2.0 - Reward-Gated Curriculum

### Features
- Added RewardGatedCurriculumSampler (now RewardCurriculumSampler)
- Single threshold design for simplicity
- Added `initial_complexity_percentile` and `final_complexity_percentile`

---

## Version 0.1.0 - Initial Release

### Features
- MultiSourceSampler with fixed weights
- CurriculumMultiSourceSampler with adaptive weights and time-based complexity
- Support for weighted and proportional sampling strategies
