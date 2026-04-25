from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class ExperimentMetadata(BaseModel):
    name: str
    version: str
    timestamp_prefix: bool = True


class HardwareConfig(BaseModel):
    gpu_sku: str
    gpu_count: int = Field(gt=0)
    vram_gb: float = Field(gt=0)

    @property
    def total_vram_gb(self) -> float:
        return self.gpu_count * self.vram_gb


class VerifierConfig(BaseModel):
    model_id: str
    dtype: str
    vram_estimate_gb: float = Field(ge=0)


class DraftConfig(BaseModel):
    model_id: str
    quantization: str
    vram_estimate_gb: float = Field(ge=0)


class CellConfig(BaseModel):
    id: str
    architecture: Literal["dense", "moe"]
    verifier: VerifierConfig
    draft: DraftConfig
    total_vram_gb: float = Field(gt=0)
    hypothesis: str = ""

    @model_validator(mode="after")
    def validate_total_vram(self) -> "CellConfig":
        component_total = self.verifier.vram_estimate_gb + self.draft.vram_estimate_gb
        if self.total_vram_gb < component_total:
            raise ValueError(
                f"total_vram_gb={self.total_vram_gb} is lower than verifier+draft "
                f"estimate {component_total} for cell {self.id}"
            )
        return self


class WorkloadConfig(BaseModel):
    id: str
    source: Literal["jsonl", "lighteval"] = "jsonl"
    data_path: str | None = None
    tasks: list[str] = Field(default_factory=list)
    num_prompts: int = Field(gt=0)
    num_prompts_per_task: int | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(gt=0)
    description: str = ""

    @model_validator(mode="after")
    def validate_source_fields(self) -> "WorkloadConfig":
        if self.source == "jsonl" and not self.data_path:
            raise ValueError(f"JSONL workload '{self.id}' requires data_path")
        if self.source == "lighteval" and not self.tasks:
            raise ValueError(f"lighteval workload '{self.id}' requires tasks")
        return self


class SweepConfig(BaseModel):
    num_speculative_tokens: list[int]
    temperatures: list[float]

    @model_validator(mode="after")
    def validate_sweep(self) -> "SweepConfig":
        if not self.num_speculative_tokens:
            raise ValueError("num_speculative_tokens must not be empty")
        if not self.temperatures:
            raise ValueError("temperatures must not be empty")
        if any(k <= 0 for k in self.num_speculative_tokens):
            raise ValueError("num_speculative_tokens values must be positive")
        return self


class BaselineConfig(BaseModel):
    run_no_speculation: bool = True
    warmup_requests: int = Field(default=10, ge=0)


class VLLMDefaults(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0)
    served_model_name: str = "benchmark-model"
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    max_model_len: int = Field(default=4096, gt=0)
    kv_cache_overhead_gb: float = Field(default=4, ge=0)
    min_vram_headroom_gb: float = Field(default=4, ge=0)
    enforce_eager: bool = False
    block_size: int = Field(default=16, gt=0)
    max_num_seqs: int = Field(default=1, gt=0)
    enable_prefix_caching: bool = False


class ExperimentConfig(BaseModel):
    experiment: ExperimentMetadata
    hardware: HardwareConfig
    model_registry: list[str]
    workload_registry: list[str]
    sweep: SweepConfig
    baselines: BaselineConfig
    vllm_defaults: VLLMDefaults
    cells: list[CellConfig] = Field(default_factory=list)
    workloads: list[WorkloadConfig] = Field(default_factory=list)


class RunConfig(BaseModel):
    run_id: str
    cell_id: str
    verifier_architecture: Literal["dense", "moe"]
    verifier_model: str
    verifier_dtype: str
    draft_model: str
    draft_quantization: str
    num_speculative_tokens: int
    temperature: float
    workload_id: str
    workload_source: Literal["jsonl", "lighteval"]
    workload_path: str | None
    workload_tasks: list[str]
    workload_declared_num_prompts: int
    workload_num_prompts_per_task: int | None
    max_output_tokens: int
    is_baseline: bool
    vllm_host: str
    vllm_port: int
    served_model_name: str
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    block_size: int
    max_num_seqs: int
    enable_prefix_caching: bool


class ExperimentPlan(BaseModel):
    config_path: str
    experiment: ExperimentMetadata
    hardware: HardwareConfig
    vllm_defaults: VLLMDefaults
    baselines: BaselineConfig
    cells: list[CellConfig]
    workloads: list[WorkloadConfig]
    runs: list[RunConfig]
    vram_warnings: list[str] = Field(default_factory=list)


def load_experiment(
    config_path: str = "configs/experiment.yaml",
    *,
    validate_vram: bool = True,
) -> ExperimentPlan:
    config_file = Path(config_path).resolve()
    raw = _load_yaml(config_file)
    cfg = ExperimentConfig.model_validate(raw)

    config_dir = config_file.parent
    project_root = config_dir.parent
    cells = [_load_cell(_resolve_path(path, config_dir, project_root)) for path in cfg.model_registry]
    workloads = [
        _load_workload(_resolve_path(path, config_dir, project_root), project_root)
        for path in cfg.workload_registry
    ]

    cfg = cfg.model_copy(update={"cells": cells, "workloads": workloads})
    warnings = check_vram_budget(cfg)
    if validate_vram and any(w.startswith("ERROR:") for w in warnings):
        detail = "\n".join(warnings)
        raise ValueError(f"VRAM validation failed for {config_file}:\n{detail}")

    return ExperimentPlan(
        config_path=str(config_file),
        experiment=cfg.experiment,
        hardware=cfg.hardware,
        vllm_defaults=cfg.vllm_defaults,
        baselines=cfg.baselines,
        cells=cells,
        workloads=workloads,
        runs=_expand_runs(cfg),
        vram_warnings=warnings,
    )


def check_vram_budget(cfg: ExperimentConfig) -> list[str]:
    messages: list[str] = []
    available = cfg.hardware.total_vram_gb
    overhead = cfg.vllm_defaults.kv_cache_overhead_gb
    min_headroom = cfg.vllm_defaults.min_vram_headroom_gb

    for cell in cfg.cells:
        required = cell.total_vram_gb + overhead
        headroom = available - required
        if headroom < 0:
            messages.append(
                "ERROR: "
                f"cell '{cell.id}' needs ~{required:.1f}GB "
                f"({cell.total_vram_gb:.1f}GB model + {overhead:.1f}GB KV/cache overhead) "
                f"but {cfg.hardware.gpu_count}x {cfg.hardware.gpu_sku} provides "
                f"~{available:.1f}GB. Reduce max_model_len, use a smaller/quantized verifier, "
                "or increase gpu_count."
            )
        elif headroom < min_headroom:
            messages.append(
                "WARNING: "
                f"cell '{cell.id}' has only ~{headroom:.1f}GB VRAM headroom. "
                "Expect KV-cache pressure on longer sequences."
            )
    return messages


def run_to_env(run: RunConfig) -> str:
    spec_config = ""
    if not run.is_baseline:
        spec_config = json.dumps(
            {
                "model": run.draft_model,
                "num_speculative_tokens": run.num_speculative_tokens,
                "quantization": run.draft_quantization,
            },
            separators=(",", ":"),
        )

    pairs = {
        "RUN_ID": run.run_id,
        "VERIFIER_MODEL": run.verifier_model,
        "VERIFIER_DTYPE": run.verifier_dtype,
        "SERVED_MODEL_NAME": run.served_model_name,
        "SPECULATIVE_CONFIG": spec_config,
        "VLLM_PORT": str(run.vllm_port),
        "GPU_MEMORY_UTILIZATION": str(run.gpu_memory_utilization),
        "MAX_MODEL_LEN": str(run.max_model_len),
        "MAX_NUM_SEQS": str(run.max_num_seqs),
        "BLOCK_SIZE": str(run.block_size),
    }
    return "\n".join(f"{key}={_shell_quote(value)}" for key, value in pairs.items())


def _expand_runs(cfg: ExperimentConfig) -> list[RunConfig]:
    runs: list[RunConfig] = []
    run_counter = 0
    defaults = cfg.vllm_defaults

    for cell in cfg.cells:
        for workload in cfg.workloads:
            if cfg.baselines.run_no_speculation:
                runs.append(
                    _build_run(
                        run_counter=run_counter,
                        run_id=f"run_{run_counter:04d}_baseline_{cell.id}_{workload.id}",
                        cell=cell,
                        workload=workload,
                        num_speculative_tokens=0,
                        temperature=0.0,
                        is_baseline=True,
                        defaults=defaults,
                    )
                )
                run_counter += 1

            for k, temp in itertools.product(
                cfg.sweep.num_speculative_tokens,
                cfg.sweep.temperatures,
            ):
                temp_label = str(temp).replace(".", "p")
                runs.append(
                    _build_run(
                        run_counter=run_counter,
                        run_id=f"run_{run_counter:04d}_{cell.id}_k{k}_t{temp_label}_{workload.id}",
                        cell=cell,
                        workload=workload,
                        num_speculative_tokens=k,
                        temperature=temp,
                        is_baseline=False,
                        defaults=defaults,
                    )
                )
                run_counter += 1

    return runs


def _build_run(
    *,
    run_counter: int,
    run_id: str,
    cell: CellConfig,
    workload: WorkloadConfig,
    num_speculative_tokens: int,
    temperature: float,
    is_baseline: bool,
    defaults: VLLMDefaults,
) -> RunConfig:
    del run_counter
    return RunConfig(
        run_id=run_id,
        cell_id=cell.id,
        verifier_architecture=cell.architecture,
        verifier_model=cell.verifier.model_id,
        verifier_dtype=cell.verifier.dtype,
        draft_model="" if is_baseline else cell.draft.model_id,
        draft_quantization="" if is_baseline else cell.draft.quantization,
        num_speculative_tokens=num_speculative_tokens,
        temperature=temperature,
        workload_id=workload.id,
        workload_source=workload.source,
        workload_path=workload.data_path,
        workload_tasks=workload.tasks,
        workload_declared_num_prompts=workload.num_prompts,
        workload_num_prompts_per_task=workload.num_prompts_per_task,
        max_output_tokens=workload.max_output_tokens,
        is_baseline=is_baseline,
        vllm_host=defaults.host,
        vllm_port=defaults.port,
        served_model_name=defaults.served_model_name,
        max_model_len=defaults.max_model_len,
        gpu_memory_utilization=defaults.gpu_memory_utilization,
        enforce_eager=defaults.enforce_eager,
        block_size=defaults.block_size,
        max_num_seqs=defaults.max_num_seqs,
        enable_prefix_caching=defaults.enable_prefix_caching,
    )


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _load_cell(path: Path) -> CellConfig:
    try:
        return CellConfig.model_validate(_load_yaml(path))
    except ValidationError as exc:
        raise ValueError(f"Invalid model config {path}: {exc}") from exc


def _load_workload(path: Path, project_root: Path) -> WorkloadConfig:
    try:
        workload = WorkloadConfig.model_validate(_load_yaml(path))
    except ValidationError as exc:
        raise ValueError(f"Invalid workload config {path}: {exc}") from exc

    if workload.source == "jsonl" and workload.data_path is not None:
        data_path = _resolve_path(workload.data_path, path.parent, project_root)
        return workload.model_copy(update={"data_path": str(data_path)})
    return workload


def _resolve_path(path: str, config_dir: Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    config_relative = config_dir / candidate
    if config_relative.exists():
        return config_relative.resolve()
    return (project_root / candidate).resolve()


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
