"""
verl_extensions package

Custom extensions for the verl framework, including:
- Multi-source data sampling with fixed or adaptive weights
- Time-based complexity curriculum
- Reward-gated complexity curriculum
"""

__version__ = "0.3.0"

# Import samplers
from .multi_source_sampler import (
    MultiSourceSampler,
    TimeCurriculumSampler,
    RewardCurriculumSampler,
    # Backwards compatibility aliases
    CurriculumMultiSourceSampler,
    RewardGatedCurriculumSampler,
)

__all__ = [
    'MultiSourceSampler',
    'TimeCurriculumSampler',
    'RewardCurriculumSampler',
    # Backwards compatibility
    'CurriculumMultiSourceSampler',
    'RewardGatedCurriculumSampler',
]
