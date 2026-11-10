"""
Configuration management for the MemQ system.

This module defines Pydantic models for configuration management,
supporting both YAML and JSON configuration files.
"""

import os
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
import yaml
import json


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = Field(default="openai", description="LLM provider name")
    api_key: str = Field(default="sk-8", 
                        description="API key for authentication")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    # Optional Azure/OpenAI-style API versioning. Some runners pass this through.
    api_version: Optional[str] = Field(default=None, description="Optional API version for some providers (e.g. Azure OpenAI)")
    model: str = Field(default="gpt-4.1-mini", description="Model name")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens for inference")
    memory_max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens for memory operations (reflection). Falls back to max_tokens if not set.")
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v


class LLMCircuitBreakerConfig(BaseModel):
    """Global service-level circuit breaker shared by all LLM providers."""

    enabled: bool = Field(default=True)
    failure_threshold: int = Field(default=3, ge=1)
    failure_window: float = Field(default=30.0, ge=0.0)
    probe_interval: float = Field(default=300.0, ge=0.0)
    confirm_interval: float = Field(default=30.0, ge=0.0)
    probe_successes: int = Field(default=2, ge=1)
    max_pause: float = Field(default=3600.0, gt=0.0)
    retry_delays: List[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0])

    @field_validator("retry_delays")
    @classmethod
    def retry_delays_must_be_non_negative(cls, values: List[float]) -> List[float]:
        if not values:
            raise ValueError("retry_delays cannot be empty")
        if any(float(value) < 0 for value in values):
            raise ValueError("retry_delays must be non-negative")
        return [float(value) for value in values]


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""
    
    provider: str = Field(default="openai", description="Embedding provider name")
    api_key: str = Field(default="sk-8",
                        description="API key for authentication")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    api_version: Optional[str] = Field(default=None, description="Optional API version for some providers (e.g. Azure OpenAI)")
    model: str = Field(default="text-embedding-3-large", description="Embedding model name")
    max_text_len: int = Field(
        default=8196,
        ge=0,
        description="Maximum characters per query before chunked embedding (0 disables chunking)",
    )
    max_token_len: int = Field(
        default=0,
        ge=0,
        description="Maximum tokens per query; truncated via tiktoken before sending (0 disables)",
    )
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v


class MemoryConfig(BaseModel):
    """Configuration for memory system."""
    
    # Strategy configuration
    build_strategy: str = Field(default="proceduralization", 
                               description="Build strategy: trajectory, script, proceduralization")
    retrieve_strategy: str = Field(default="query",
                                  description="Retrieve strategy: random, query, avefact") 
    update_strategy: str = Field(default="adjustment",
                                description="Update strategy: vanilla, validation, adjustment")
    
    # Memory parameters
    k_retrieve: int = Field(default=1, gt=-1, description="Number of memories to retrieve")
    max_keywords: int = Field(default=8, gt=0, description="Maximum keywords for AveFact")
    add_similarity_threshold: float = Field(default=0.9,
                                    description="similarity_threshold for add memory")
    user_id: str = Field(default="memq_user", description="User ID for memory management")
    max_memory_images: int = Field(default=10, ge=0, description="Max images from retrieved memories to include in prompt (0=no limit)")

    # Optional checkpoint loading for runners that support resuming memory state.
    load_from_checkpoint: bool = Field(default=False, description="Whether to load memory state from a checkpoint snapshot")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to a memory checkpoint snapshot to load (if enabled)")


class EnvironmentConfig(BaseModel):
    """Configuration for environment-specific settings."""
    pass


class ExperimentConfig(BaseModel):
    """Configuration for experimental settings."""

    # Experiment parameters
    experiment_name: str = Field(..., description="Name for the experiment trail")
    algorithm: str = Field(
        default="memq",
        description=(
            "Algorithm to run: memq, memrl, self_rag, rag, memp"
        ),
    )
    memrl_update_mode: str = Field(
        default="batch",
        description="When to apply direct Q-value updates: 'batch' (per mini-batch) or 'epoch' (deferred to epoch end)",
    )
    memq_update_mode: str = Field(
        default="epoch",
        description="When to flush eligibility-trace ΔQ: 'batch' (per mini-batch) or 'epoch' (deferred to epoch end)",
    )
    val_before_train: bool = Field(
        default=True, description="Whether to run validation before training starts"
    )
    enable_value_driven: bool = Field(default=False, description="Whether to use rl")
    random_seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )
    mode: str = Field(
        default="train", description="Control whether to only train or test"
    )

    # Optional tracing (LLB JSONL) controlled by YAML (YAML overrides env vars when set).
    trace_jsonl_path: Optional[str] = Field(
        default=None,
        description="If set, enable LLB per-task JSONL tracing and write to this path.",
    )
    trace_sample_filter: Optional[str] = Field(
        default=None,
        description="Optional task filter for tracing: digits=first N tasks; or comma-separated sample_index list.",
    )

    # Q-value normalization toggle used in hybrid retrieval scoring.
    normalize_q: bool = Field(
        default=True,
        description=(
            "Max-scale normalize Q values (q / q_max) before hybrid scoring. "
            "Set false to use raw Q values."
        ),
    )
    llb_q_floor: Optional[float] = Field(
        default=None,
        description=(
            "LLB only: if set, overrides rl_config.q_floor for LLB runs. "
            "Mimics memory_rl's q_floor to prevent Q from dropping below this value "
            "(applied during both Q update and retrieval scoring)."
        ),
    )
    llb_dedup_by_task_id: bool = Field(
        default=False,
        description=(
            "LLB only: when selecting the final top-k retrieved memories, deduplicate by task_id "
            "(fallback to sample_index/id; if still missing, treated as unique). "
            "This mimics memory_rl's task-level Phase-B behavior and reduces same-task repeats."
        ),
    )

    # LLB-specific parameters
    task: str = Field(default="db", description="Task type for LLB: db, os, kg")
    split_file: str = Field(default="", description="Path to LLB dataset split file")
    valid_file: Optional[str] = Field(
        default=None, description="Path to LLB validation dataset file"
    )

    num_epochs: int = Field(default=5, description="Number of epochs to train")
    batch_size: int = Field(default=32, description="Number of parallel environments for sampling")
    max_steps: int = Field(default=30, description="Max steps per episode during training and evaluation")
    valid_interval: int = Field(default=1, description="Run evaluation on the validation set every N epochs. Set to 0 to disable.")
    dataset_ratio: float = Field(default=0.7, description="Proportion of files randomly selected for training (rest used for validation)")
    # DSBench data-analysis settings
    dsbench_root: Optional[str] = Field(
        default=None,
        description="Path to the official DSBench checkout with data_analysis/data",
    )
    dsbench_docker_image: str = Field(
        default="memq-dsbench:py310",
        description="Docker image used to execute generated DSBench Python code",
    )
    dsbench_executor_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout in seconds for each generated Python execution",
    )
    dsbench_max_output_chars: int = Field(
        default=8000,
        ge=512,
        description="Maximum captured stdout/stderr characters per execution",
    )
    dsbench_judge_max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum output tokens for the DSBench semantic judge",
    )
    self_rag_planner_max_tokens: int = Field(
        default=4096,
        ge=4096,
        description="Initial output-token budget for the Self-RAG controller",
    )
    self_rag_reranker_max_tokens: int = Field(
        default=8192,
        ge=8192,
        description="Initial output-token budget for the Self-RAG critic/reranker",
    )
    self_rag_max_retries: int = Field(
        default=1,
        ge=0,
        le=4,
        description="Retries for truncated or invalid Self-RAG JSON responses",
    )
    self_rag_log_max_chars: int = Field(
        default=2000,
        ge=0,
        le=10000,
        description="Maximum characters in one Self-RAG prompt-summary log line",
    )
    dsbench_task_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries for transient per-task DSBench inference errors",
    )
    dsbench_task_retry_backoff: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
        description="Base exponential backoff in seconds for DSBench task retries",
    )
    dsbench_max_train_tasks: int = Field(
        default=0,
        ge=0,
        description="Optional train-task cap for smoke tests (0 means all)",
    )
    dsbench_max_eval_tasks: int = Field(
        default=0,
        ge=0,
        description="Optional validation-task cap for smoke tests (0 means all)",
    )
    dsbench_train_task_ids: list[str] = Field(
        default_factory=list,
        description="Optional exact DSBench sample IDs to use for training",
    )
    dsbench_eval_task_ids: list[str] = Field(
        default_factory=list,
        description="Optional exact DSBench sample IDs to use for validation",
    )
    few_shot_path: str = Field(default='', description="Path for few-shot examples")
    bon: int = Field(default=0, description="Run BoN-evaluation on the val/test for N trails")
    # GPQA-specific parameters
    gpqa_dataset: str = Field(default="diamond", description="GPQA dataset variant: diamond, main, or extended")
    gpqa_prompt_type: str = Field(default="chain_of_thought", description="GPQA prompt type: zero_shot, chain_of_thought")
    gpqa_data_path: Optional[str] = Field(default=None, description="Path to GPQA CSV data directory")
    gpqa_use_few_shot: bool = Field(default=False, description="GPQA CoT: prepend 5-shot examples from official benchmark (default False)")
    train_valid_split: float = Field(default=0.8, description="Ratio to split training and validation sets")
    baseline_mode: Optional[str] = Field(default=None, description="Baseline mode: passk or reflection")
    baseline_k: int = Field(default=10, description="Baseline rounds (k) for pass@k/reflection")
    # Output settings
    output_dir: str = Field(default="./results", description="Directory for experiment outputs")
    save_trajectories: bool = Field(default=True, description="Save detailed trajectories")
    save_memories: bool = Field(default=True, description="Save memory snapshots")

    # Transfer learning: strip detailed trajectory from success memories, keeping only the high-level script.
    strip_success_trajectory: bool = Field(
        default=False,
        description=(
            "Strip the TRAJECTORY section from successful memories before prompt injection. "
            "Keeps only the SCRIPT (high-level strategy). Useful for transfer learning "
            "where task-specific trajectories don't generalize to new test sets."
        ),
    )

    # Logging settings
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    log_level: str = Field(default="INFO", description="Logging level")
    logger: Literal["tensorboard", "swanlab"] = Field(
        default="tensorboard",
        description="Experiment metric logger backend",
    )
    swanlab_project: str = Field(
        default="MemQ",
        min_length=1,
        description="SwanLab project name when logger='swanlab'",
    )
    swanlab_workspace: Optional[str] = Field(
        default=None,
        description="SwanLab workspace username when logger='swanlab'",
    )

class RLConfig(BaseModel):
    """Configuration for reinforcement learning parameters."""

    epsilon: float = Field(default=0.1, description="ε-greedy exploration probability")
    tau: float = Field(default=0.35, description="Unknown detection threshold on similarity")
    alpha: float = Field(default=0.3, description="Q-learning step size (learning rate)")
    gamma: float = Field(default=0.0, description="Discount factor (default 0 for single-step)")
    lambd: float = Field(default=0.8, description="Lambda for TD(lambda) eligibility traces")
    q_init_pos: float = Field(default=0.0, description="Optimistic initialization for positive Q-values")
    q_init_neg: float = Field(default=0.0, description="Initialization for negative Q-values")
    q_floor: Optional[float] = Field(
        default=None,
        description=(
            "Minimum allowed Q value (optional). "
            "Only applied by runners that explicitly enable it (e.g., LLB via experiment.llb_q_floor)."
        ),
    )
    q_ceiling: Optional[float] = Field(
        default=None,
        description="Maximum Q value (clamp ceiling). Used by memq algorithm.",
    )
    success_reward: float = Field(default=1.0, description="Reward for successful outcome")
    failure_reward: float = Field(default=-1.0, description="Reward for failure outcome")
    # Retrieval filtering threshold used by runners when calling MemoryService.retrieve_query(...).
    # (Kept separate from `tau` to avoid conflating unknown-detection vs retrieval filtering.)
    sim_threshold: float = Field(default=0.5, description="Similarity threshold for retrieval filtering")
    topk: int = Field(default=5, description="Candidate set size for value-aware selection")
    novelty_threshold: float = Field(default=0.85, description="Similarity threshold to treat as non-novel (merge)")
    q_min_threshold: float = Field(default=-0.8, description="Threshold for q min")
    weight_sim: float = Field(default=0.5, description="Weight for similarity in combined score")
    weight_q: float = Field(default=0.5, description="Weight for Q-value in combined score")
    structural_max_depth: int = Field(
        default=7,
        ge=1,
        description="Maximum BFS traversal depth for structural trace decay",
    )
    normalize_dq: bool = Field(
        default=False,
        description="Normalize accumulated memq ΔQ by contribution count before flushing",
    )
    q_init_adaptive: bool = Field(
        default=False,
        description="Initialize new memory Q as mean Q of retrieved memories instead of fixed q_init",
    )
    alpha_min: float = Field(default=0.0, description="Minimum alpha at end of cosine annealing schedule")
    alpha_schedule: str = Field(default="constant", description="Alpha schedule: 'constant' or 'cosine'")


class MemQConfig(BaseModel):
    """Main configuration class for the MemQ system."""

    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    llm_circuit_breaker: LLMCircuitBreakerConfig = Field(
        default_factory=LLMCircuitBreakerConfig
    )
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    rl_config: RLConfig = Field(default_factory=RLConfig)
    # Global settings
    project_name: str = Field(default="memq", description="Project name")
    version: str = Field(default="1.0.0", description="Project version")

    model_config = ConfigDict(extra="forbid")  # Don't allow extra fields
        
    @classmethod
    def from_yaml(cls, config_path: str) -> "MemQConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            MemQConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
                
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    @classmethod
    def from_json(cls, config_path: str) -> "MemQConfig":
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            MemQConfig instance
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    def to_yaml(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, output_path: str, indent: int = 2) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save JSON configuration
            indent: JSON indentation level
        """
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=indent)
    
    def get_strategy_config(self):
        """Get strategy configuration for MemoryService."""
        from ..service.strategies import StrategyConfiguration
        
        return StrategyConfiguration.from_strings(
            build=self.memory.build_strategy,
            retrieve=self.memory.retrieve_strategy,
            update=self.memory.update_strategy
        )
    
    def validate_paths(self) -> None:
        """
        Validate that all specified paths exist.

        Raises:
            FileNotFoundError: If required paths don't exist
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        strategy_str = f"{self.memory.build_strategy}+{self.memory.retrieve_strategy}+{self.memory.update_strategy}"
        return f"MemQConfig(strategy={strategy_str}, llm={self.llm.model}, embedding={self.embedding.model})"
