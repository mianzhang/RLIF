#!/usr/bin/env python3
"""
Script to generate training recipes for RLIF experiments.
Generates slurm scripts with different model sizes, datasets, and hyperparameters.

Usage:
    python generate_train_recipes.py
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from constant import (
    QWEN3_06B, QWEN3_17B, QWEN3_4B, QWEN3_8B,
    MIX_IF1000_LOGICIF1000, MIX_IF80000,
    MIX_IF20000_LOGICIF60000, MIX_IF40000_LOGICIF40000,
    MIX_IF60000_LOGICIF20000, MIX_LOGICIF80000,
    RULEIFEVAL
)


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""
    # Required fields
    model: any
    train_mixture: any
    val_mixture: any
    
    # Training parameters
    train_batch_size: int = 512
    learning_rate: float = 1e-6
    total_epochs: int = 10
    total_training_steps: int = 500
    
    # PPO parameters
    ppo_mini_batch_size: int = 128
    ppo_micro_batch_size: int = 16
    
    # Sequence lengths
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    
    # Hardware configuration
    n_gpus: int = 8
    nodes: int = 1
    
    # Checkpoint and validation
    save_freq: int = 50
    test_freq: int = 25
    
    # Behavior flags
    enable_thinking: bool = False
    slurm: bool = True
    time_limit: str = "5-00:00:00"
    
    def get_job_name(self) -> str:
        """Generate job name from configuration."""
        thinking_suffix = "_think" if self.enable_thinking else "_nothink"
        return f"{self.model.nickname}{thinking_suffix}_{self.train_mixture.nickname}_grpo"
    
    def get_filename(self) -> str:
        """Generate filename from configuration."""
        thinking_suffix = "_think" if self.enable_thinking else "_nothink"
        slurm_prefix = "slurm_" if self.slurm else ""
        return f"{slurm_prefix}{self.model.nickname}{thinking_suffix}_{self.train_mixture.nickname}_grpo.sh"


def generate_slurm_header(config: TrainingConfig) -> list[str]:
    """Generate SLURM header lines."""
    job_name = config.get_job_name()
    return [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --gpus-per-node={config.n_gpus}",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --exclusive",
        f"#SBATCH --time={config.time_limit}",
        f"#SBATCH --output=log/{job_name}.log",
        "",
        "",
    ]


def generate_training_command(config: TrainingConfig) -> list[str]:
    """Generate the main training command lines."""
    job_name = config.get_job_name()
    log_file = f"log/{job_name}.log"
    cuda_devices = ",".join(str(i) for i in range(config.n_gpus))
    
    return [
        "export $(grep -v '^#' .env | xargs)",
        f"export CUDA_VISIBLE_DEVICES={cuda_devices}",
        f"n_gpus_per_node={config.n_gpus}",
        f"MODEL_PATH={config.model.model_path}",
        "",
        "python3 -m verl.trainer.main_ppo \\",
        "    algorithm.adv_estimator=grpo \\",
        f"    data.train_files={config.train_mixture.data_file} \\",
        f"    data.val_files={config.val_mixture.data_file} \\",
        f"    data.train_batch_size={config.train_batch_size} \\",
        f"    data.max_prompt_length={config.max_prompt_length} \\",
        f"    data.max_response_length={config.max_response_length} \\",
        "    data.filter_overlong_prompts=True \\",
        "    data.truncation='error' \\",
        f"    +data.apply_chat_template_kwargs.enable_thinking={'True' if config.enable_thinking else 'False'} \\",
        "    actor_rollout_ref.model.path=$MODEL_PATH \\",
        f"    actor_rollout_ref.actor.optim.lr={config.learning_rate} \\",
        "    actor_rollout_ref.model.use_remove_padding=True \\",
        f"    actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size} \\",
        f"    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.ppo_micro_batch_size} \\",
        "    actor_rollout_ref.actor.use_kl_loss=True \\",
        "    actor_rollout_ref.actor.kl_loss_coef=0.001 \\",
        "    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\",
        "    actor_rollout_ref.actor.entropy_coeff=0 \\",
        "    actor_rollout_ref.actor.strategy=fsdp2 \\",
        "    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \\",
        "    actor_rollout_ref.model.enable_gradient_checkpointing=False \\",
        "    actor_rollout_ref.actor.fsdp_config.param_offload=True \\",
        "    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\",
        "    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \\",
        "    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\",
        "    actor_rollout_ref.rollout.name=vllm \\",
        "    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\",
        "    actor_rollout_ref.rollout.n=8 \\",
        "    actor_rollout_ref.rollout.dtype=bfloat16 \\",
        "    actor_rollout_ref.rollout.max_num_batched_tokens=10255 \\",
        "    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \\",
        "    actor_rollout_ref.ref.fsdp_config.param_offload=True \\",
        "    actor_rollout_ref.ref.strategy=fsdp2 \\",
        "    algorithm.use_kl_in_reward=False \\",
        "    trainer.critic_warmup=0 \\",
        "    trainer.logger=['console','wandb'] \\",
        "    trainer.project_name='RLIF' \\",
        f"    trainer.experiment_name='{job_name}' \\",
        "    trainer.n_gpus_per_node=$n_gpus_per_node \\",
        f"    trainer.nnodes={config.nodes} \\",
        f"    trainer.save_freq={config.save_freq} \\",
        "    trainer.val_before_train=True \\",
        "    trainer.val_only=False \\",
        "    trainer.resume_mode=disable \\",
        "    trainer.resume_from_path=null \\",
        f"    trainer.test_freq={config.test_freq} \\",
        f"    trainer.total_epochs={config.total_epochs}\\",
        f"    trainer.total_training_steps={config.total_training_steps} > {log_file}",
        "",
    ]


def generate_script(config: TrainingConfig) -> str:
    """Generate complete training script from configuration."""
    lines = []
    
    # Add SLURM header if requested
    if config.slurm:
        lines.extend(generate_slurm_header(config))
    else:
        lines.extend(["", "#!/bin/bash"])
    
    # Add training command
    lines.extend(generate_training_command(config))
    
    return "\n".join(lines)


def create_output_directories():
    """Create necessary output directories."""
    Path("recipe/rlif").mkdir(parents=True, exist_ok=True)
    Path("log").mkdir(exist_ok=True)


def save_script(config: TrainingConfig, output_dir: Path) -> Path:
    """Save generated script to file and make it executable."""
    filepath = output_dir / config.get_filename()
    script_content = generate_script(config)
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    os.chmod(filepath, 0o755)
    return filepath


def define_training_configs() -> list[TrainingConfig]:
    """
    Define all training configurations.
    Override defaults only where needed for clarity.
    """

    return [
        # Small model, small dataset - with thinking
        TrainingConfig(
            model=QWEN3_06B,
            train_mixture=MIX_IF1000_LOGICIF1000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=True,
            save_freq=4,
            total_epochs=2
        ),
        
        # Small model, small dataset - without thinking
        TrainingConfig(
            model=QWEN3_06B,
            train_mixture=MIX_IF1000_LOGICIF1000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=False,
            save_freq=4,
            total_epochs=2
        ),

        # Qwen3-8B Models (Stage 1) 
        TrainingConfig(
            model=QWEN3_8B,
            train_mixture=MIX_IF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=True,
            save_freq=100,
        ),

        TrainingConfig(
            model=QWEN3_8B,
            train_mixture=MIX_IF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=False,
            save_freq=100,
        ),

        TrainingConfig(
            model=QWEN3_8B,
            train_mixture=MIX_LOGICIF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=True,
            save_freq=100,
        ),

        TrainingConfig(
            model=QWEN3_8B,
            train_mixture=MIX_LOGICIF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=False,
            save_freq=100,
        ),

        # Qwen3-1.7B Models (Stage 1) 
        TrainingConfig(
            model=QWEN3_17B,
            train_mixture=MIX_IF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=True,
            save_freq=100,
        ),

        TrainingConfig(
            model=QWEN3_17B,
            train_mixture=MIX_IF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=False,
            save_freq=100,
        ),

        TrainingConfig(
            model=QWEN3_17B,
            train_mixture=MIX_LOGICIF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=True,
            save_freq=100,
        ),

        TrainingConfig(
            model=QWEN3_17B,
            train_mixture=MIX_LOGICIF80000,
            val_mixture=RULEIFEVAL,
            train_batch_size=512,
            enable_thinking=False,
            save_freq=100,
        ),
    ]


def print_summary(generated_files: list[str], output_dir: Path):
    """Print summary of generated files."""
    print(f"\n{'='*60}")
    print(f"Successfully generated {len(generated_files)} training recipes")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for filename in generated_files:
        print(f"  ✓ {filename}")
    print("\nUsage:")
    print(f"  # Run SLURM job:")
    print(f"  sbatch {output_dir}/<script_name>.sh")
    print(f"  # Run directly:")
    print(f"  bash {output_dir}/<script_name>.sh")


def main():
    """Generate all training recipe scripts."""
    # Setup
    create_output_directories()
    output_dir = Path("recipe/rlif")
    
    # Define configurations
    configs = define_training_configs()
    
    # Generate and save scripts
    generated_files = []
    for config in configs:
        filepath = save_script(config, output_dir)
        generated_files.append(config.get_filename())
        print(f"Generated: {filepath}")
    
    # Print summary
    print_summary(generated_files, output_dir)


if __name__ == "__main__":
    main() 