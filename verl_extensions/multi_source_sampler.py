"""Multi-source sampler for verl framework.

Provides samplers for training on multiple data sources with different strategies:
- MultiSourceSampler: Base sampler with fixed or adaptive source weights
- TimeCurriculumSampler: Time-based complexity curriculum
- RewardCurriculumSampler: Reward-gated complexity curriculum
"""

import numpy as np
from collections import defaultdict
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class MultiSourceSampler(AbstractCurriculumSampler):
    """
    Samples indices to create batches with mixed data sources.
    
    Each batch will contain data from multiple sources according to
    the configured sampling strategy and weights.
    
    Config:
        sampling_strategy: str
            - 'weighted': Use source_weights for custom ratios
            - 'proportional': Weight by dataset size
        source_weights: dict
            Mapping of {data_source_name: weight}
        adaptive_weights: bool (default: False)
            If True, dynamically adjust weights based on source rewards.
            Sources with lower rewards get higher weights (focus on harder tasks).
        
        # Adaptive weights parameters (only used if adaptive_weights=True):
        alpha: float (0-1)
            EMA smoothing factor. Higher = more reactive. Default: 0.1
        update_frequency: int
            Update weights every N batches. Default: 5
        temperature: float (>0)
            Softmax temperature. Lower = more sensitive. Default: 1.0
        max_source_ratio: float (>=1)
            Maximum ratio between highest and lowest weight. Default: 3.0
    
    Example:
        # Fixed weights (1:1 ratio)
        sampler:
          class_name: MultiSourceSampler
          sampling_strategy: weighted
          source_weights:
            source1: 1.0
            source2: 1.0
        
        # Adaptive weights
        sampler:
          class_name: MultiSourceSampler
          sampling_strategy: weighted
          source_weights:
            source1: 1.0
            source2: 1.0
          adaptive_weights: true
          alpha: 0.1
          update_frequency: 5
          temperature: 0.3
          max_source_ratio: 3.0
    """
    
    def __init__(self, data_source, data_config: DictConfig):
        self.dataset = data_source
        self.batch_size = data_config.train_batch_size
        self.seed = data_config.get('seed', 1)
        self.epoch = 0
        
        # Get sampler config
        sampler_cfg = data_config.sampler
        self.sampling_strategy = sampler_cfg.get('sampling_strategy', 'weighted')
        self.source_weights = dict(sampler_cfg.get('source_weights', {}))
        
        # Adaptive weights configuration
        self.adaptive_weights = sampler_cfg.get('adaptive_weights', False)
        if self.adaptive_weights:
            self.alpha = sampler_cfg.get('alpha', 0.1)
            self.update_frequency = sampler_cfg.get('update_frequency', 5)
            self.temperature = sampler_cfg.get('temperature', 1.0)
            self.max_source_ratio = sampler_cfg.get('max_source_ratio', 3.0)
        
        # Build source mapping
        self.source_to_indices = self._build_source_mapping()
        self.sources = sorted(self.source_to_indices.keys())
        
        if not self.sources:
            raise ValueError("No data sources found in dataset!")
        
        # Validate configuration
        self._validate()
        
        # Compute initial weights
        self.normalized_weights = self._compute_weights()
        self.rng = np.random.RandomState(self.seed)
        
        # Initialize adaptive weight tracking if enabled
        if self.adaptive_weights:
            self._init_adaptive_weights()
        
        # Log initialization
        self._log_initialization()
    
    def _validate(self):
        """Validate configuration."""
        valid_strategies = ['weighted', 'proportional']
        if self.sampling_strategy not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {valid_strategies}")
        
        if self.adaptive_weights:
            if not 0 < self.alpha <= 1:
                raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
            if self.temperature <= 0:
                raise ValueError(f"temperature must be > 0, got {self.temperature}")
            if self.max_source_ratio < 1:
                raise ValueError(f"max_source_ratio must be >= 1, got {self.max_source_ratio}")
    
    def _init_adaptive_weights(self):
        """Initialize adaptive weight tracking."""
        self.reward_ema = {s: 0.0 for s in self.sources}
        self.initialized = {s: False for s in self.sources}
        self.batch_count = 0
    
    def _log_initialization(self):
        """Log sampler initialization."""
        mode = "adaptive" if self.adaptive_weights else "fixed"
        print(f"\n[MultiSourceSampler] Strategy: {self.sampling_strategy}, "
              f"Mode: {mode}, Sources: {len(self.sources)}, Batch size: {self.batch_size}")
        
        for src, count in sorted([(s, len(self.source_to_indices[s])) for s in self.sources]):
            print(f"  {src}: {count} samples (weight: {self.normalized_weights[src]:.3f})")
        
        if self.adaptive_weights:
            min_weight = 1.0 / (1.0 + self.max_source_ratio)
            max_weight = self.max_source_ratio / (1.0 + self.max_source_ratio)
            print(f"[MultiSourceSampler] Adaptive: alpha={self.alpha}, update_freq={self.update_frequency}, "
                  f"temp={self.temperature}")
            print(f"[MultiSourceSampler] Weight bounds: [{min_weight:.1%}, {max_weight:.1%}]")
    
    def _build_source_mapping(self):
        """Build mapping from data_source to list of indices."""
        source_to_indices = defaultdict(list)
        for idx in range(len(self.dataset)):
            source = self.dataset[idx]['data_source']
            source_to_indices[source].append(idx)
        return dict(source_to_indices)
    
    def _compute_weights(self):
        """Compute normalized weights for each source."""
        if self.sampling_strategy == 'weighted':
            weights = {s: self.source_weights.get(s, 1.0) for s in self.sources}
        elif self.sampling_strategy == 'proportional':
            weights = {s: len(self.source_to_indices[s]) for s in self.sources}
        
        total = sum(weights.values())
        return {s: w / total for s, w in weights.items()}
    
    def _get_source_indices(self, source):
        """Get indices for a source. Override in subclasses for filtering."""
        return self.source_to_indices[source]
    
    def _sample_batch(self):
        """Sample indices for one batch with all sources mixed."""
        indices = []
        allocated = 0
        
        for i, source in enumerate(self.sources[:-1]):
            n_samples = int(self.batch_size * self.normalized_weights[source])
            source_indices = self._get_source_indices(source)
            sampled = self.rng.choice(source_indices, n_samples, replace=True)
            indices.extend([int(idx) for idx in sampled])
            allocated += n_samples
        
        # Last source gets remaining samples
        remaining = self.batch_size - allocated
        last_source = self.sources[-1]
        source_indices = self._get_source_indices(last_source)
        sampled = self.rng.choice(source_indices, remaining, replace=True)
        indices.extend([int(idx) for idx in sampled])
        
        self.rng.shuffle(indices)
        return indices
    
    def __iter__(self):
        """Generate indices for the epoch."""
        self.rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        
        num_batches = len(self.dataset) // self.batch_size
        
        # For adaptive/curriculum samplers, generate lazily
        if self.adaptive_weights or isinstance(self, AbstractCurriculumSampler):
            def lazy_generator():
                for _ in range(num_batches):
                    yield from self._sample_batch()
            return lazy_generator()
        else:
            # Pre-generate all indices (more efficient)
            indices = []
            for _ in range(num_batches):
                indices.extend(self._sample_batch())
            return iter(indices)
    
    def __len__(self):
        return (len(self.dataset) // self.batch_size) * self.batch_size
    
    # ========== Adaptive Weight Methods ==========
    
    def update(self, batch) -> None:
        """Update method called after each batch. Only used with adaptive_weights."""
        if not self.adaptive_weights:
            return
        
        if 'token_level_scores' not in batch.batch:
            raise ValueError("token_level_scores not found in batch")
        
        # Aggregate rewards by source
        rewards = batch.batch['token_level_scores'].sum(-1).cpu().numpy()
        sources = batch.non_tensor_batch['data_source']
        
        source_rewards = defaultdict(list)
        for reward, source in zip(rewards, sources):
            source_rewards[source].append(float(reward))
        
        # Update EMA for each source
        for source, reward_list in source_rewards.items():
            if source not in self.reward_ema:
                continue
            
            avg_reward = np.mean(reward_list)
            
            if not self.initialized[source]:
                self.reward_ema[source] = avg_reward
                self.initialized[source] = True
            else:
                self.reward_ema[source] = (
                    self.alpha * avg_reward + (1 - self.alpha) * self.reward_ema[source]
                )
        
        self.batch_count += 1
        
        # Periodically adjust weights
        if self.batch_count % self.update_frequency == 0:
            self._adjust_adaptive_weights()
    
    def _adjust_adaptive_weights(self):
        """Adjust source weights based on reward EMAs."""
        if not all(self.initialized.values()):
            print(f"[MultiSourceSampler] Batch {self.batch_count}: Waiting for all sources")
            return
        
        # Inverse relationship: lower reward → higher weight
        neg_emas = [-self.reward_ema[s] / self.temperature for s in self.sources]
        max_neg = max(neg_emas)
        exp_vals = [np.exp(x - max_neg) for x in neg_emas]
        total = sum(exp_vals)
        
        raw_weights = {s: exp_val / total for s, exp_val in zip(self.sources, exp_vals)}
        
        # Apply ratio bound
        min_raw = min(raw_weights.values())
        max_raw = max(raw_weights.values())
        
        if max_raw / min_raw > self.max_source_ratio:
            target_min = 1.0 / (1.0 + self.max_source_ratio)
            target_max = self.max_source_ratio / (1.0 + self.max_source_ratio)
            
            clipped_weights = {}
            for s, w in raw_weights.items():
                clipped_weights[s] = max(target_min, min(target_max, w))
            
            total_weight = sum(clipped_weights.values())
            final_weights = {s: w / total_weight for s, w in clipped_weights.items()}
        else:
            final_weights = raw_weights
        
        self.source_weights.update({s: float(w) for s, w in final_weights.items()})
        self.normalized_weights = self._compute_weights()
        
        # Log update
        weights_list = [self.normalized_weights[s] for s in self.sources]
        actual_ratio = max(weights_list) / min(weights_list)
        
        print(f"\n[MultiSourceSampler] Batch {self.batch_count} - Weight adjustment:")
        for src in self.sources:
            ema = self.reward_ema[src]
            weight = self.normalized_weights[src]
            print(f"  {src}: reward_ema={ema:6.2f}, weight={weight:.3f} ({weight*100:.1f}%)")
        print(f"  Actual max/min ratio: {actual_ratio:.2f}x")
    
    def get_stats(self):
        """Get current sampler statistics."""
        stats = {
            'sources': {
                src: {
                    'weight': self.normalized_weights[src],
                    'count': len(self.source_to_indices[src])
                }
                for src in self.sources
            }
        }
        
        if self.adaptive_weights:
            weights_list = [self.normalized_weights[s] for s in self.sources]
            stats['adaptive'] = {
                'enabled': True,
                'batch_count': self.batch_count,
                'reward_ema': dict(self.reward_ema),
                'max_min_ratio': max(weights_list) / min(weights_list) if min(weights_list) > 0 else float('inf')
            }
        
        return stats


# ============================================================================
# CURRICULUM SAMPLER BASE
# ============================================================================

class _CurriculumSamplerMixin:
    """Mixin providing common complexity curriculum functionality."""
    
    def _extract_complexity_data(self, source):
        """Extract and sort complexity data for a source."""
        complexity_data = []
        for idx in self.source_to_indices[source]:
            item = self.dataset[idx]
            complexity = None
            if 'extra_info' in item and isinstance(item['extra_info'], dict):
                complexity = item['extra_info'].get('complexity_score')
            elif 'complexity_score' in item:
                complexity = item['complexity_score']
            
            if complexity is not None:
                complexity_data.append((idx, float(complexity)))
        
        if not complexity_data:
            raise ValueError(f"No complexity scores found for '{source}'")
        
        # Sort by complexity (ascending: easier first)
        complexity_data.sort(key=lambda x: x[1])
        return complexity_data


# ============================================================================
# TIME-BASED CURRICULUM SAMPLER
# ============================================================================

class TimeCurriculumSampler(MultiSourceSampler, _CurriculumSamplerMixin):
    """
    Time-based curriculum sampler that gradually increases data complexity.
    
    Strategy: Start with easiest X% of data, linearly increase to Y% over
    a fixed number of training steps (time-based progression).
    
    Also supports adaptive source weights (inherited from MultiSourceSampler).
    
    Config:
        # Inherited from MultiSourceSampler
        sampling_strategy, source_weights, adaptive_weights, etc.
        
        # Time-based curriculum config
        complexity_source: str
            Which data source to apply curriculum to. Default: 'logicif'
        initial_complexity_percentile: int (0-100)
            Start with easiest X% of data. Default: 30
        final_complexity_percentile: int (0-100)
            Eventually use up to Y% of data. Default: 100
        complexity_warmup_steps: int
            Number of weight updates to reach final complexity. Default: 100
    
    Example:
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
    """
    
    def __init__(self, data_source, data_config: DictConfig):
        # Initialize base sampler
        super().__init__(data_source, data_config)
        
        cfg = data_config.sampler
        self.complexity_source = cfg.get('complexity_source', 'logicif')
        self.initial_complexity_percentile = cfg.get('initial_complexity_percentile', 30)
        self.final_complexity_percentile = cfg.get('final_complexity_percentile', 100)
        self.complexity_warmup_steps = cfg.get('complexity_warmup_steps', 100)
        
        # Validate
        if not 0 < self.initial_complexity_percentile < 100:
            raise ValueError(f"initial_complexity_percentile must be in (0, 100)")
        if not 0 < self.final_complexity_percentile <= 100:
            raise ValueError(f"final_complexity_percentile must be in (0, 100]")
        if self.initial_complexity_percentile >= self.final_complexity_percentile:
            raise ValueError("initial must be < final complexity percentile")
        
        # Initialize curriculum state
        self.update_count = 0
        self.current_complexity_percentile = self.initial_complexity_percentile
        self.complexity_indices = {}
        
        # Build complexity indices
        if self.complexity_source in self.sources:
            self._build_complexity_indices()
        else:
            print(f"[TimeCurriculumSampler] Warning: complexity source '{self.complexity_source}' not found")
        
        # Log
        print(f"\n[TimeCurriculumSampler] Complexity curriculum for '{self.complexity_source}'")
        print(f"[TimeCurriculumSampler] Progress: {self.initial_complexity_percentile}% → "
              f"{self.final_complexity_percentile}% over {self.complexity_warmup_steps} updates")
    
    def _build_complexity_indices(self):
        """Build sorted indices by complexity score."""
        source = self.complexity_source
        complexity_data = self._extract_complexity_data(source)
        
        self.complexity_indices[source] = {
            'sorted_indices': [idx for idx, _ in complexity_data],
            'complexity_scores': [score for _, score in complexity_data]
        }
        
        min_score = complexity_data[0][1]
        max_score = complexity_data[-1][1]
        median_score = complexity_data[len(complexity_data)//2][1]
        print(f"[TimeCurriculumSampler] Complexity stats: min={min_score:.3f}, "
              f"median={median_score:.3f}, max={max_score:.3f}, n={len(complexity_data)}")
    
    def _get_source_indices(self, source):
        """Get indices filtered by current complexity percentile."""
        if source != self.complexity_source or source not in self.complexity_indices:
            return self.source_to_indices[source]
        
        sorted_indices = self.complexity_indices[source]['sorted_indices']
        cutoff_idx = max(1, int(len(sorted_indices) * self.current_complexity_percentile / 100.0))
        filtered = sorted_indices[:cutoff_idx]
        
        if len(filtered) < 10:
            filtered = sorted_indices[:min(10, len(sorted_indices))]
        
        return filtered
    
    def update(self, batch) -> None:
        """Update adaptive weights and complexity threshold."""
        # Update adaptive weights (from base class)
        super().update(batch)
        
        # Update complexity threshold
        self._update_complexity_threshold()
    
    def _update_complexity_threshold(self):
        """Gradually increase complexity threshold."""
        self.update_count += 1
        
        progress = min(1.0, self.update_count / self.complexity_warmup_steps)
        old_percentile = self.current_complexity_percentile
        self.current_complexity_percentile = (
            self.initial_complexity_percentile + 
            progress * (self.final_complexity_percentile - self.initial_complexity_percentile)
        )
        
        # Log significant changes
        if int(old_percentile / 10) != int(self.current_complexity_percentile / 10):
            if self.complexity_source in self.complexity_indices:
                sorted_indices = self.complexity_indices[self.complexity_source]['sorted_indices']
                cutoff_idx = int(len(sorted_indices) * self.current_complexity_percentile / 100.0)
                print(f"[TimeCurriculumSampler] Complexity: {self.current_complexity_percentile:.0f}% "
                      f"({cutoff_idx}/{len(sorted_indices)} items)")
    
    def get_stats(self):
        """Get current curriculum statistics."""
        stats = super().get_stats()
        stats['curriculum'] = {
            'type': 'time',
            'complexity_source': self.complexity_source,
            'current_percentile': self.current_complexity_percentile,
            'initial_percentile': self.initial_complexity_percentile,
            'final_percentile': self.final_complexity_percentile,
            'warmup_progress': min(1.0, self.update_count / self.complexity_warmup_steps)
        }
        return stats


# ============================================================================
# REWARD-GATED CURRICULUM SAMPLER
# ============================================================================

class RewardCurriculumSampler(MultiSourceSampler, _CurriculumSamplerMixin):
    """
    Reward-gated curriculum sampler that unlocks harder data based on performance.
    
    Strategy: Start with easiest X% of data, divide remaining data into N intervals.
    Unlock next interval when model achieves the reward threshold.
    
    Also supports adaptive source weights (inherited from MultiSourceSampler).
    
    Config:
        # Inherited from MultiSourceSampler
        sampling_strategy, source_weights, adaptive_weights, etc.
        
        # Reward-gated curriculum config
        complexity_source: str
            Which data source to apply curriculum to. Default: 'logicif'
        initial_complexity_percentile: int (0-100)
            Start with easiest X% of data. Default: 30
        final_complexity_percentile: int (0-100)
            Maximum data to use (exclude hardest). Default: 100
        num_intervals: int
            Number of intervals between initial and final. Default: 10
        reward_threshold: float
            Reward EMA threshold to unlock next interval. Required.
        reward_alpha: float (0-1)
            EMA smoothing for reward tracking. Default: 0.1
        min_batches_per_level: int
            Minimum batches before checking threshold. Default: 10
        check_frequency: int
            Check threshold every N batches. Default: 3
        focus_new_intervals: bool
            If True, newly unlocked intervals have higher sampling weight.
            Default: False
        interval_decay: float (0-1)
            Decay factor for older intervals when focus_new_intervals=True.
            Weight of interval i = interval_decay ^ (current_interval - i)
            Default: 0.5 (each older interval has half the weight)
    
    Example:
        sampler:
          class_name: RewardCurriculumSampler
          sampling_strategy: weighted
          source_weights:
            iftrain: 1.0
            logicif: 1.0
          complexity_source: logicif
          initial_complexity_percentile: 30
          final_complexity_percentile: 100
          num_intervals: 10
          reward_threshold: 0.125
          reward_alpha: 0.1
          min_batches_per_level: 10
          check_frequency: 3
          focus_new_intervals: true
          interval_decay: 0.5
    """
    
    def __init__(self, data_source, data_config: DictConfig):
        # Initialize base sampler
        super().__init__(data_source, data_config)
        
        cfg = data_config.sampler
        self.complexity_source = cfg.get('complexity_source', 'logicif')
        self.initial_complexity_percentile = cfg.get('initial_complexity_percentile', 30)
        self.final_complexity_percentile = cfg.get('final_complexity_percentile', 100)
        self.num_intervals = cfg.get('num_intervals', 10)
        self.reward_threshold = float(cfg.get('reward_threshold', 0.5))
        self.reward_alpha = cfg.get('reward_alpha', cfg.get('alpha', 0.1))
        self.min_batches_per_level = cfg.get('min_batches_per_level', 10)
        self.check_frequency = cfg.get('check_frequency', 3)
        
        # Focus on new intervals parameters
        self.focus_new_intervals = cfg.get('focus_new_intervals', False)
        self.interval_decay = cfg.get('interval_decay', 0.5)
        
        # Validate
        if self.num_intervals < 1:
            raise ValueError(f"num_intervals must be >= 1")
        if not 0 < self.initial_complexity_percentile < 100:
            raise ValueError(f"initial_complexity_percentile must be in (0, 100)")
        if not 0 < self.final_complexity_percentile <= 100:
            raise ValueError(f"final_complexity_percentile must be in (0, 100]")
        if self.initial_complexity_percentile >= self.final_complexity_percentile:
            raise ValueError("initial must be < final complexity percentile")
        if 'reward_threshold' not in cfg:
            raise ValueError("reward_threshold is required")
        if self.focus_new_intervals and not (0 < self.interval_decay <= 1):
            raise ValueError(f"interval_decay must be in (0, 1], got {self.interval_decay}")
        
        # Initialize curriculum state
        self.current_interval = 0
        self.curriculum_reward_ema = 0.0
        self.curriculum_reward_initialized = False
        self.curriculum_batch_count = 0
        self.batches_at_current_level = 0
        self.complexity_intervals = {}
        
        # Build complexity intervals
        if self.complexity_source in self.sources:
            self._build_complexity_intervals()
        else:
            raise ValueError(f"Complexity source '{self.complexity_source}' not found")
        
        # Log
        print(f"\n[RewardCurriculumSampler] Complexity source: '{self.complexity_source}'")
        print(f"[RewardCurriculumSampler] Initial: {self.initial_complexity_percentile}%, "
              f"Final: {self.final_complexity_percentile}%")
        print(f"[RewardCurriculumSampler] Intervals: {self.num_intervals + 1} "
              f"(1 initial + {self.num_intervals} progressive)")
        print(f"[RewardCurriculumSampler] Threshold: {self.reward_threshold}, "
              f"alpha: {self.reward_alpha}")
        if self.focus_new_intervals:
            print(f"[RewardCurriculumSampler] Focus on new intervals: decay={self.interval_decay}")
    
    def _build_complexity_intervals(self):
        """Build intervals from initial to final percentile."""
        source = self.complexity_source
        complexity_data = self._extract_complexity_data(source)
        
        total_samples = len(complexity_data)
        initial_cutoff = int(total_samples * self.initial_complexity_percentile / 100.0)
        final_cutoff = int(total_samples * self.final_complexity_percentile / 100.0)
        
        # Interval 0: Initial pool (easiest X%)
        initial_data = complexity_data[:initial_cutoff]
        if initial_data:
            indices = [idx for idx, _ in initial_data]
            scores = [score for _, score in initial_data]
            self.complexity_intervals[0] = {
                'indices': indices,
                'min_complexity': min(scores),
                'max_complexity': max(scores),
                'count': len(indices)
            }
            print(f"  Interval 0 (Initial {self.initial_complexity_percentile}%): "
                  f"{len(indices)} samples, [{min(scores):.3f}, {max(scores):.3f}]")
        
        # Progressive intervals
        progressive_data = complexity_data[initial_cutoff:final_cutoff]
        if progressive_data and self.num_intervals > 0:
            samples_per = len(progressive_data) / self.num_intervals
            
            for i in range(1, self.num_intervals + 1):
                start = int((i - 1) * samples_per)
                end = int(i * samples_per) if i < self.num_intervals else len(progressive_data)
                
                interval_data = progressive_data[start:end]
                if not interval_data:
                    continue
                
                indices = [idx for idx, _ in interval_data]
                scores = [score for _, score in interval_data]
                
                self.complexity_intervals[i] = {
                    'indices': indices,
                    'min_complexity': min(scores),
                    'max_complexity': max(scores),
                    'count': len(indices)
                }
                print(f"  Interval {i}: {len(indices)} samples, [{min(scores):.3f}, {max(scores):.3f}]")
        
        # Log excluded data
        if final_cutoff < total_samples:
            excluded = complexity_data[final_cutoff:]
            excluded_scores = [s for _, s in excluded]
            print(f"  [Excluded {100-self.final_complexity_percentile}%]: "
                  f"{len(excluded)} samples, [{min(excluded_scores):.3f}, {max(excluded_scores):.3f}]")
    
    def _get_source_indices(self, source):
        """Get indices from unlocked intervals, optionally with recency weighting."""
        if source != self.complexity_source:
            return self.source_to_indices[source]
        
        # If focus_new_intervals is disabled, just combine all unlocked intervals
        if not self.focus_new_intervals:
            unlocked = []
            for i in range(self.current_interval + 1):
                if i in self.complexity_intervals:
                    unlocked.extend(self.complexity_intervals[i]['indices'])
            
            if len(unlocked) < 10 and 0 in self.complexity_intervals:
                unlocked = self.complexity_intervals[0]['indices']
            
            return unlocked
        
        # Focus on new intervals: weight by recency
        # Weight of interval i = interval_decay ^ (current_interval - i)
        # Newest interval (current_interval) has weight 1.0
        # Older intervals have exponentially decaying weights
        
        weighted_indices = []
        total_weight = 0.0
        
        for i in range(self.current_interval + 1):
            if i not in self.complexity_intervals:
                continue
            
            # Calculate weight: newer intervals have higher weight
            age = self.current_interval - i  # 0 for newest, increases for older
            weight = self.interval_decay ** age
            total_weight += weight
            
            # Store interval indices with their weight
            weighted_indices.append((i, self.complexity_intervals[i]['indices'], weight))
        
        if not weighted_indices:
            if 0 in self.complexity_intervals:
                return self.complexity_intervals[0]['indices']
            return self.source_to_indices[source]
        
        # Normalize weights
        normalized_weights = [(i, indices, w / total_weight) for i, indices, w in weighted_indices]
        
        # Create a weighted sample pool
        # We'll over-sample newer intervals by repeating their indices
        # Use a sampling multiplier based on weight
        max_count = max(len(indices) for _, indices, _ in normalized_weights)
        base_multiplier = 10  # Base repetition factor
        
        weighted_pool = []
        for interval_idx, indices, weight in normalized_weights:
            # Repeat indices proportional to their weight
            repeats = max(1, int(weight * base_multiplier * len(normalized_weights)))
            for _ in range(repeats):
                weighted_pool.extend(indices)
        
        return weighted_pool
    
    def update(self, batch) -> None:
        """Update adaptive weights and check for level unlock."""
        # Update adaptive weights (from base class)
        super().update(batch)
        
        # Update curriculum reward tracking
        self._update_curriculum_reward(batch)
        
        # Check for level unlock
        self.curriculum_batch_count += 1
        self.batches_at_current_level += 1
        
        if self.curriculum_batch_count % self.check_frequency == 0:
            self._check_and_unlock()
    
    def _update_curriculum_reward(self, batch):
        """Update curriculum reward EMA."""
        if 'token_level_scores' not in batch.batch:
            return
        
        rewards = batch.batch['token_level_scores'].sum(-1).cpu().numpy()
        avg_reward = float(np.mean(rewards))
        
        if not self.curriculum_reward_initialized:
            self.curriculum_reward_ema = avg_reward
            self.curriculum_reward_initialized = True
        else:
            self.curriculum_reward_ema = (
                self.reward_alpha * avg_reward + 
                (1 - self.reward_alpha) * self.curriculum_reward_ema
            )
    
    def _check_and_unlock(self):
        """Check if should unlock next interval."""
        # Already at max
        if self.current_interval >= self.num_intervals:
            return
        
        # Need minimum batches
        if self.batches_at_current_level < self.min_batches_per_level:
            return
        
        # Check threshold
        if self.curriculum_reward_ema >= self.reward_threshold:
            old = self.current_interval
            self.current_interval += 1
            self.batches_at_current_level = 0
            
            old_count = sum(self.complexity_intervals[i]['count'] 
                          for i in range(old + 1) if i in self.complexity_intervals)
            new_count = sum(self.complexity_intervals[i]['count'] 
                          for i in range(self.current_interval + 1) if i in self.complexity_intervals)
            
            info = self.complexity_intervals.get(self.current_interval, {})
            
            print(f"\n{'='*80}")
            print(f"[RewardCurriculumSampler] LEVEL UP! {old} → {self.current_interval}")
            print(f"[RewardCurriculumSampler] Batch {self.curriculum_batch_count}: "
                  f"Reward {self.curriculum_reward_ema:.3f} >= {self.reward_threshold:.3f}")
            if info:
                print(f"[RewardCurriculumSampler] Unlocked: [{info['min_complexity']:.3f}, "
                      f"{info['max_complexity']:.3f}]")
            print(f"[RewardCurriculumSampler] Samples: {old_count} → {new_count}")
            
            # Show interval weights if focus_new_intervals is enabled
            if self.focus_new_intervals:
                weight_info = []
                total_w = sum(self.interval_decay ** (self.current_interval - i) 
                             for i in range(self.current_interval + 1) 
                             if i in self.complexity_intervals)
                for i in range(self.current_interval + 1):
                    if i in self.complexity_intervals:
                        w = (self.interval_decay ** (self.current_interval - i)) / total_w
                        weight_info.append(f"I{i}:{w:.1%}")
                print(f"[RewardCurriculumSampler] Interval weights: {', '.join(weight_info)}")
            
            if self.current_interval < self.num_intervals:
                print(f"[RewardCurriculumSampler] Next unlock at reward >= {self.reward_threshold:.3f}")
            else:
                print(f"[RewardCurriculumSampler] All intervals unlocked!")
            print(f"{'='*80}\n")
        else:
            # Log progress periodically
            if self.curriculum_batch_count % (self.check_frequency * 5) == 0:
                progress = (self.curriculum_reward_ema / self.reward_threshold) * 100
                print(f"[RewardCurriculumSampler] Batch {self.curriculum_batch_count}: "
                      f"Level {self.current_interval}/{self.num_intervals}, "
                      f"Reward {self.curriculum_reward_ema:.3f} ({progress:.1f}% of threshold)")
    
    def get_stats(self):
        """Get current curriculum statistics."""
        stats = super().get_stats()
        
        unlocked = sum(self.complexity_intervals[i]['count'] 
                      for i in range(self.current_interval + 1) 
                      if i in self.complexity_intervals)
        total = sum(v['count'] for v in self.complexity_intervals.values())
        
        stats['curriculum'] = {
            'type': 'reward',
            'complexity_source': self.complexity_source,
            'current_interval': self.current_interval,
            'max_intervals': self.num_intervals + 1,
            'reward_ema': self.curriculum_reward_ema,
            'reward_threshold': self.reward_threshold,
            'unlocked_samples': unlocked,
            'total_samples': total,
            'completion_ratio': unlocked / total if total > 0 else 0,
            'focus_new_intervals': self.focus_new_intervals
        }
        
        # Add interval weights if focus_new_intervals is enabled
        if self.focus_new_intervals and self.current_interval >= 0:
            total_w = sum(self.interval_decay ** (self.current_interval - i) 
                         for i in range(self.current_interval + 1) 
                         if i in self.complexity_intervals)
            if total_w > 0:
                stats['curriculum']['interval_weights'] = {
                    i: (self.interval_decay ** (self.current_interval - i)) / total_w
                    for i in range(self.current_interval + 1)
                    if i in self.complexity_intervals
                }
        
        return stats


# ============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ============================================================================

# Keep old names as aliases for backwards compatibility
CurriculumMultiSourceSampler = TimeCurriculumSampler
RewardGatedCurriculumSampler = RewardCurriculumSampler
