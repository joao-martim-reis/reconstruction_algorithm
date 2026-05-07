# copilot.md — Python Engineering Standards

> **Scope:** This file governs all code written in this project — by humans and AI alike.
> **Domain:** Python — machine learning, deep learning, data science, and scientific computing pipelines.
> **AI Usage:** GitHub Copilot assists development. Every suggestion is a draft. Every rule here applies without exception.
> **Living document:** Update this file when new patterns are adopted or old ones are retired.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [AI-Assisted Coding Guidelines](#2-ai-assisted-coding-guidelines)
3. [Project Structure](#3-project-structure)
4. [Naming Conventions](#4-naming-conventions)
5. [Comments & Documentation](#5-comments--documentation)
6. [Logging & Experiment Tracking](#6-logging--experiment-tracking)
7. [Modularity & Functions](#7-modularity--functions)
8. [Error Handling](#8-error-handling)
9. [Data Pipelines](#9-data-pipelines)
10. [Model & Algorithm Design](#10-model--algorithm-design)
11. [Types & Validation](#11-types--validation)
12. [Testing Standards](#12-testing-standards)
13. [Reproducibility](#13-reproducibility)
14. [Performance](#14-performance)
15. [Security & Data Governance](#15-security--data-governance)
16. [Configuration & Environment](#16-configuration--environment)
17. [Dependencies](#17-dependencies)
18. [Version Control & Commits](#18-version-control--commits)
19. [Code Review Checklist](#19-code-review-checklist)
20. [Tooling](#20-tooling)
21. [Key Principles](#21-key-principles)

---

## 1. Core Philosophy

- **Clarity over cleverness** — Code is read far more than it is written. A dense one-liner that needs a paragraph to explain is not clever — it is a liability.
- **Reproducibility is correctness** — A result that cannot be reproduced is not a valid result. Every experiment must be traceable to a seed, a config, a dataset, and a commit.
- **Explicit over implicit** — Every assumption — shapes, units, value ranges, data types — must be stated in code or in a comment. Hidden conventions break silently.
- **Modular by default** — Data loading, preprocessing, modelling, training, and evaluation are separate concerns. They must be independently testable and replaceable.
- **Fail fast** — Validate inputs at every boundary. Crash loudly on bad data. Silent failures in numerical pipelines produce wrong results with no error.
- **Consistency** — Follow established patterns throughout the project. Propose changes through review rather than introducing local exceptions.
- **Measure before optimising** — Profile before rewriting. Do not add complexity for performance without data that justifies it.

---

## 2. AI-Assisted Coding Guidelines

Copilot accelerates development. It does not replace domain knowledge or engineering judgment.

- **Read every suggestion before accepting.** Copilot does not know your data, your constraints, or your numerical requirements.
- **Write the docstring and type signature first.** A precise function signature produces better completions than an empty body.
- **Verify numerical code manually.** Loss functions, normalisations, and metrics are frequently plausible-looking but subtly wrong.
- **Never let Copilot finalise variable or function names** without checking them against the naming rules in Section 4.
- **If you cannot explain it, do not ship it.** Understand every accepted suggestion before committing it.
- **Simplify generated code.** If a suggestion is correct but dense, rewrite it for readability before committing.
- **All rules in this document apply to Copilot output.** AI-generated code is not exempt.

---

## 3. Project Structure

```
project/
├── src/
│   ├── data/            # Dataset classes, loaders, transforms
│   ├── models/          # Model and algorithm definitions only — no training logic
│   ├── pipelines/       # End-to-end orchestration — calls other modules, no logic of its own
│   ├── evaluation/      # Metrics, evaluation loops, result visualisation
│   ├── inference/       # Inference-only pipeline, stripped of training dependencies
│   ├── config/          # Configuration schemas and defaults
│   └── utils/           # Shared, stateless helper functions
├── tests/
│   ├── unit/            # Unit tests — mirrors src/ structure
│   ├── integration/     # Pipeline-level tests using small synthetic data
│   └── fixtures/        # Shared test data, synthetic inputs, factory functions
├── notebooks/           # Exploratory analysis only — never production code
├── scripts/             # One-off automation scripts
├── configs/             # YAML experiment configs — no hardcoded values in src/
├── docs/                # Architecture notes, dataset cards, model cards
├── .env.example         # Documented template for all required environment variables
├── pyproject.toml       # Single source of truth for dependencies and tooling
├── copilot.md           # This file
└── README.md            # Project overview and setup guide
```

**Rules:**
- A new team member must understand the layout within 5 minutes.
- `models/` contains architecture and algorithm definitions only — no training loops, no data loading.
- `pipelines/` orchestrates — it calls functions from other modules and contains no algorithm logic itself.
- `notebooks/` is a scratchpad. Code that needs to run reliably must live in `src/` and have tests.
- Every module exposes a clean public API through its `__init__.py`. Never import from deep internal paths outside the module.

---

## 4. Naming Conventions

Follow PEP 8 without exception.

### General Rules

| Thing | Convention | Example |
|---|---|---|
| Variables and functions | `snake_case` | `learning_rate`, `compute_loss` |
| Classes | `PascalCase` | `ImageDataset`, `UNetModel` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_EPOCHS`, `DEFAULT_LR` |
| Modules and files | `snake_case` | `data_loader.py`, `train_utils.py` |
| Booleans | Prefix with `is_`, `has_`, `use_`, `should_` | `is_training`, `use_amp` |
| Collections | Plural nouns | `image_paths`, `batch_losses` |

### Be Descriptive — No Abbreviations

```python
# ✓ Clear and specific
reconstruction_error = compute_error(prediction, target)
validation_image_paths = load_paths(split="val")

# ✗ Cryptic — meaningless outside the author's head
recon_err = compute_error(pred, tgt)
val_imgs = load_paths(split="val")
```

### Name Arrays and Tensors by Their Content and Shape

In numerical and ML code, a variable name must tell a reader what the data represents — not just that it is "some array". Include the shape context when it aids clarity.

```python
# ✓ Shape-aware and content-aware
image_batch          # a batch of images, shape (B, C, H, W)
class_probabilities  # per-class softmax outputs, shape (B, num_classes)
sinogram             # projection data, shape (angles, detectors)

# ✗ Shape-blind — the reader has no idea what this contains
x
data
out
tmp
inp
arr
```

### Functions

Start with a verb. The name must describe what the function does.

```python
# ✓
def compute_validation_loss(...)
def load_dataset(...)
def apply_augmentation(...)
def save_checkpoint(...)

# ✗
def loss(...)       # noun — what does it do?
def dataset(...)    # noun
def augment(...)    # too vague — augment what?
```

---

## 5. Comments & Documentation

> **Rule:** Code shows *what* is happening. Comments explain *why* — including domain decisions, mathematical reasoning, and non-obvious numerical choices.

### Comment the Why, Not the What

```python
# ✓ Explains the reasoning — adds value the code alone cannot
# Clamp outputs to [0, 1] before computing SSIM.
# SSIM is undefined outside this range and returns misleadingly low values
# for visually correct reconstructions that have minor out-of-range drift.
output = output.clamp(0.0, 1.0)

# ✗ Restates the code — zero value
# Clamp output between 0 and 1
output = output.clamp(0.0, 1.0)
```

### Docstrings — Google Style, Required on All Public Functions and Classes

```python
def train_one_epoch(model, loader, optimiser, loss_fn, device):
    """Run a single training epoch and return the mean loss.

    Args:
        model: The model in training mode. Caller must call model.train() beforehand.
        loader: DataLoader yielding batches as dicts with "input" and "target" keys.
        optimiser: Configured optimiser. Scheduler is managed by the caller.
        loss_fn: Callable with signature (prediction, target) -> scalar tensor.
        device: Compute device. All tensors will be moved here before the forward pass.

    Returns:
        Mean loss over all batches as a Python float.

    Raises:
        RuntimeError: If a non-finite loss is encountered and cannot be recovered.
    """
```

- Include `Args`, `Returns`, and `Raises` for every public function.
- Add an `Example:` block whenever the usage is non-obvious.
- For functions that accept or return arrays or tensors, document the expected shape and value range in the argument description.

### Marker Tags

Always include a description. A bare marker is not acceptable.

```
# TODO(name):   Work required before the next release. Link the ticket.
# FIXME(name):  Known defect. Describe the symptom and the impact.
# HACK(name):   Temporary workaround. Explain why it exists and how to remove it.
# NOTE:         Context that all future maintainers must understand.
# MATH:         Non-trivial operation — include the formula or a reference.
```

### What Not to Do

```python
# ✗ Commented-out dead code — delete it, Git history preserves everything
# result = old_function(x)

# ✗ Execution trace comments
# reached this point
# step 2 done

# ✗ Debug print statements left in code
print(x.shape)
print("here")
```

---

## 6. Logging & Experiment Tracking

### Never Use `print()` in Pipeline Code

```python
# ✗ Unstructured, not filterable, disappears in production
print(f"epoch {epoch}, loss: {loss:.4f}")

# ✓ Structured logger — filterable by level, consistent format
import logging
logger = logging.getLogger(__name__)

logger.info("Epoch complete", extra={"epoch": epoch, "loss": loss, "lr": current_lr})
logger.warning("NaN detected in loss", extra={"epoch": epoch, "batch": batch_idx})
logger.error("Checkpoint save failed", extra={"path": str(path), "reason": str(e)})
```

### Log Levels

| Level | When to use |
|---|---|
| `ERROR` | Pipeline cannot continue. Immediate attention required. |
| `WARNING` | Anomaly occurred but execution continued (NaN, unexpected range, skipped sample). |
| `INFO` | Meaningful state transition: run started, dataset loaded, checkpoint saved. |
| `DEBUG` | Detailed diagnostics for development. Disabled in production. |

### Experiment Tracking

Log all hyperparameters, metrics, and artefacts to a tracker (MLflow, Weights & Biases, or equivalent). Never rely on console output as the sole record of an experiment.

- **Log the full config at the start of every run.**
- **Log metrics at every epoch** with consistent key names (`train/loss`, `val/psnr`).
- **Log the Git commit hash** for every run. A result without a code reference cannot be reproduced.
- **Log all saved artefacts** (checkpoints, plots, result files) so the run is self-contained.

### Logging Rules

- Never log inside a tight inner loop — log aggregated statistics per epoch or per N steps.
- Never log raw arrays or tensors — log scalar values, shapes, or summary statistics.
- Never log sensitive data — file paths containing PII, patient identifiers, credentials.
- Log anomalies immediately: NaN/Inf in loss or gradients, out-of-range values, skipped samples.

---

## 7. Modularity & Functions

### One Function, One Job

```python
# ✗ One function doing data loading, preprocessing, training, evaluation, and saving
def run(config):
    data = [(img / 255, label) for img, label in load_raw(config.path)]
    model = build_model(config)
    for epoch in range(config.epochs):
        for img, label in data:
            loss = model(img, label)
            loss.backward()
    torch.save(model.state_dict(), "model.pt")
    print(sum(evaluate(model, img, label) for img, label in data) / len(data))

# ✓ Each concern is its own function — independently testable and replaceable
def build_dataset(path, transform):   ...
def build_model(cfg):                 ...
def train_one_epoch(model, loader):   ...
def evaluate(model, loader, metrics): ...
def save_checkpoint(model, path):     ...

def run_pipeline(cfg):
    # Orchestrator only — no logic, just sequencing
    dataset   = build_dataset(cfg.data.path, build_transform(cfg))
    model     = build_model(cfg.model)
    for epoch in range(cfg.training.epochs):
        train_loss   = train_one_epoch(model, dataset.train_loader)
        val_metrics  = evaluate(model, dataset.val_loader, METRICS)
        save_checkpoint(model, cfg.checkpoint_dir / f"epoch_{epoch}.pt")
```

### Function Length and Parameters

- **Aim for under 30 lines.** If a function is growing, extract named sub-functions.
- **Maximum 4 positional parameters.** For anything more complex, use a `dataclass` or config object — it is self-documenting at the call site and avoids silent argument-order mistakes.

```python
# ✗ Easy to silently swap arguments
def train(model, loader, lr, wd, epochs, device, amp, clip):
    ...

# ✓ Named, typed, and validated at construction
@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    use_amp: bool = True
    grad_clip_norm: float = 1.0

def train(model, loader, cfg: TrainingConfig):
    ...
```

### Guard Clauses — Validate Early, Avoid Nesting

```python
# ✗ Happy path buried under nested conditions
def process(data, config):
    if data is not None:
        if data.ndim == 3:
            if not has_nans(data):
                return _run(data, config)
    return None

# ✓ Each failure is named and explicit; happy path is obvious
def process(data, config):
    if data is None:
        raise ValueError("data must not be None")
    if data.ndim != 3:
        raise ValueError(f"expected 3D array, got shape {data.shape}")
    if has_nans(data):
        raise ValueError("data contains NaN values")
    return _run(data, config)
```

### Prefer Pure Functions

A pure function takes inputs and returns outputs with no side effects. It is always easier to test, easier to reason about, and safe to parallelise.

---

## 8. Error Handling

### Never Silently Swallow Exceptions

```python
# ✗ Hides failures — the pipeline continues in a corrupted state
try:
    result = run_computation(inputs)
except Exception:
    pass

# ✓ Handle deliberately — log context, recover explicitly, or re-raise
try:
    result = run_computation(inputs)
except KnownRecoverableError as e:
    logger.warning("Computation failed, using fallback", extra={"reason": str(e)})
    result = fallback_value
except Exception as e:
    logger.error("Unexpected failure", extra={"input_id": input_id})
    raise RuntimeError(f"Computation failed for input '{input_id}'") from e
```

### Define Domain-Specific Exceptions

```python
# src/exceptions.py
class PipelineError(Exception):
    """Base class for all pipeline errors."""

class DataValidationError(PipelineError):
    """Input data failed shape, dtype, or value range validation."""

class ConvergenceError(PipelineError):
    """An iterative solver did not converge within the allowed iterations."""

class CheckpointError(PipelineError):
    """A model checkpoint could not be loaded or saved."""
```

Using specific exception types allows callers to handle known failure modes precisely instead of catching `Exception` everywhere.

### Validate at Every Boundary

Check shapes, types, value ranges, and for NaN/Inf before any computation begins — not halfway through it.

```python
def fit(data, labels, config):
    if data.shape[0] != labels.shape[0]:
        raise DataValidationError(
            f"data and labels must have the same number of samples, "
            f"got {data.shape[0]} and {labels.shape[0]}"
        )
    if has_nans(data):
        raise DataValidationError("data contains NaN values — check the preprocessing pipeline")
    # All inputs validated — safe to proceed
    ...
```

---

## 9. Data Pipelines

- **Separate loading from preprocessing.** A dataset class loads and indexes raw samples. Transforms handle preprocessing. Composing them in a `DataLoader` is the pipeline's job.
- **All transforms must be stateless and composable.** A transform takes a sample and returns a transformed sample. It must not depend on external state.
- **Normalise with dataset-level statistics, not per-sample.** Always compute statistics on the training split only and store them in the config. Never use per-sample min/max normalisation in a model pipeline.
- **Do not apply training augmentations to validation or test splits.** Augmentation must be conditional on the split.
- **Document the data state at every stage** — the dtype, shape, and value range after each transform. Put this in comments or docstrings.
- **Validate samples before returning them.** A `Dataset.__getitem__` must check that the sample it returns is valid. Log and skip corrupted samples — never return garbage silently.

---

## 10. Model & Algorithm Design

- **Model files define architecture only.** No data loading, no training loops, no evaluation code. The model boundary is a `forward()` method or a `fit()` / `predict()` interface.
- **Document input and output contracts** — expected shape, dtype, and value range — in the class docstring. This is the contract all callers depend on.
- **Validate inputs in the forward method.** A clear error at the model boundary is far better than a cryptic shape error three layers deep.
- **Every iterative algorithm must have** a maximum iteration limit, a convergence criterion, and a residual or loss history returned to the caller.
- **All tunable parameters must come from config.** Regularisation strengths, learning rates, thresholds — nothing is hardcoded.
- **Prefer small, composable building blocks** over monolithic architectures. Each block must be independently testable.
- **Save only the state dictionary, not the entire model object.** Whole-object saves break across refactors. Always include the config alongside the weights.

---

## 11. Types & Validation

### Type Annotations Are Required

All function signatures must include type annotations. Use `from __future__ import annotations` for forward references.

```python
from __future__ import annotations
from pathlib import Path

def load_checkpoint(path: Path, device: str = "cpu") -> dict:
    """Load a model checkpoint from disk."""

def compute_metric(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute a scalar metric between prediction and target arrays."""
```

### Use `dataclasses` for Structured Config and Results

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 16
    use_amp: bool = True

    def __post_init__(self) -> None:
        # Validate at construction — fail before any compute starts
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
```

### Runtime Validation

TypeScript types are erased at runtime. The same is true in Python — type hints are not enforced at execution. Validate data explicitly:

- Check shapes and dtypes before passing arrays into functions.
- Check for NaN/Inf in inputs and outputs of numerical operations.
- Check value ranges where the domain demands it (e.g. probabilities must be in [0, 1]).

---

## 12. Testing Standards

### File Conventions

- Unit tests mirror `src/` exactly: `src/models/unet.py` → `tests/unit/models/test_unet.py`
- Integration tests live in `tests/integration/` and test pipeline stages with small synthetic data.
- Fixtures (synthetic arrays, mock configs, factory functions) live in `tests/fixtures/`.

### Structure: Arrange — Act — Assert

```python
def test_normalise_maps_values_to_unit_range():
    # Arrange
    data = np.array([0.0, 50.0, 100.0])
    mean, std = 50.0, 25.0

    # Act
    result = normalise(data, mean=mean, std=std)

    # Assert
    assert result.min() >= -3.0   # within 3 std of zero
    assert result.max() <=  3.0
    assert result.dtype == np.float32
```

### Coverage Requirements

| Layer | Minimum |
|---|---|
| Pure utilities | 100% |
| Core algorithms and operators | 90% |
| Model architectures | 80% — shape and forward-pass tests required |
| Pipeline orchestrators | 70% — integration tests cover the rest |

### Testing Rules

- **Deterministic.** Fix all seeds before any test that uses random data.
- **Isolated.** Unit tests use synthetic data only. Never depend on real datasets or network calls.
- **Test contracts, not internals.** Tests must not break when you refactor implementation details without changing behaviour.
- **Test failure paths.** Every `raise` in production code must have a corresponding `pytest.raises` test.
- **Test numerical properties.** For algorithms: convergence, known outputs on synthetic inputs, boundary conditions.
- **One clear assertion focus per test.** A failing test must point to exactly one thing.

---

## 13. Reproducibility

- **Fix all random seeds** at the start of every script — Python's `random`, `numpy`, and framework-specific seeds (e.g. `torch.manual_seed`). The seed must come from config, never generated randomly.
- **Log the full resolved config** at the start of every run.
- **Log the Git commit hash** for every run. A result without a code reference cannot be reproduced.
- **Log dataset name, version, and split sizes** at run start.
- **Checkpoints must include the config** that produced them alongside the weights.
- **No hardcoded values anywhere in `src/`.** Every parameter lives in a config file.
- **Deterministic operations where possible.** Be explicit about any setting that trades determinism for speed.

---

## 14. Performance

Profile before optimising. The following are the most common actual bottlenecks — address them in order before touching model code.

- **Data loading is usually the bottleneck.** Check GPU or CPU utilisation. If it is below 80%, the pipeline is starving the compute — fix the data loader first.
- **Use multiple workers in your data loader.** Tune `num_workers` to the number of available CPU cores.
- **Cache expensive deterministic operations.** If a preprocessing step produces the same output for the same input every time, compute it once and cache the result.
- **Parallelize independent operations.** Do not serialise work that can run concurrently.
- **Use mixed precision training** where supported. It halves memory usage and increases throughput with negligible accuracy impact.
- **Select only the data you need.** Avoid loading entire datasets into memory when streaming is sufficient.
- **Never optimise without a measurement.** Add a profiler result or benchmark comparison to any PR that adds complexity for performance reasons.

---

## 15. Security & Data Governance

- **Never commit patient data, sensitive data, or any PII to version control.** Use `.gitignore` aggressively. Enforce with a pre-commit hook that blocks large binary files.
- **Never commit credentials or secrets.** API keys, passwords, and tokens belong in environment variables only.
- **Anonymise before logging.** Never log identifiers, file paths containing names, or metadata that could identify individuals.
- **Document data access controls** in `docs/data_governance.md` — who is authorised to access the data and under what conditions.
- **Use paths from config, not from user input.** Constructing file paths from unchecked input is a directory traversal risk.
- **Audit dependencies regularly** for known vulnerabilities. Block merges on high-severity findings.

---

## 16. Configuration & Environment

### .env.example — Document Every Variable

```bash
# Compute
CUDA_VISIBLE_DEVICES=0

# Paths
DATA_ROOT=/data/your-dataset
CHECKPOINT_DIR=/outputs/checkpoints
RESULTS_DIR=/outputs/results

# Experiment tracking
WANDB_API_KEY=replace-with-your-key
MLFLOW_TRACKING_URI=http://localhost:5000

# External storage (optional)
S3_BUCKET_NAME=your-bucket
```

### Rules

- All environment-specific values live in `.env` — never hardcoded in `src/`.
- Validate all required environment variables at startup. Crash immediately with a clear message if anything is missing.
- Commit `.env.example` with every variable documented. Never commit `.env`.
- Load and validate config in one central location so misconfiguration is caught before any compute begins.

---

## 17. Dependencies

- **Pin all versions.** Use exact pins in `pyproject.toml` and commit the lock file.
- **Audit before adding.** Every new dependency adds maintenance burden and attack surface. Prefer the standard library or an already-present package over adding a new one for a simple task.
- **Separate runtime from development dependencies.** Linters, test frameworks, and notebooks tools must not be installed in production or inference environments.
- **Review changelogs before upgrading** core scientific libraries. NumPy, SciPy, and PyTorch minor versions frequently change numerical behaviour in ways that affect results.
- **Run a vulnerability scanner in CI.** Block merges on high-severity findings.

```toml
# pyproject.toml
[project.dependencies]
numpy      = "==1.26.4"
torch      = "==2.3.1"
hydra-core = "==1.3.2"
mlflow     = "==2.13.0"

[project.optional-dependencies]
dev = [
    "pytest==8.2.0",
    "pytest-cov==5.0.0",
    "ruff==0.4.4",
    "mypy==1.10.0",
    "pre-commit==3.7.0",
]
```

---

## 18. Version Control & Commits

### What Never to Commit

```
.env                         # Real credentials
data/                        # Raw or processed datasets
*.npy / *.h5 / *.pt / *.pth  # Large binary files — use a data/artefact registry
outputs/ / results/          # Experiment outputs — tracked in MLflow or W&B
__pycache__/ / *.pyc         # Python bytecode
notebooks/*.ipynb            # Only commit with all outputs cleared
```

### Branch Naming

```
feature/short-description       New functionality
fix/short-description           Bug fix
experiment/short-description    Exploratory model or algorithm change
refactor/short-description      Code restructuring, no behaviour change
docs/short-description          Documentation only
chore/short-description         Maintenance, tooling, dependencies
```

### Commit Message Format

```
<type>(<scope>): <short summary — present tense, max 72 chars>

[optional body — explain WHY, not what changed]

[optional footer — Closes #issue, BREAKING CHANGE:]
```

**Types:** `feat`, `fix`, `experiment`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

```
# ✓ Clear and traceable
feat(training): add gradient clipping to prevent exploding gradients

Without clipping, training on noisy batches occasionally diverged after
epoch 30. Clipping at norm=1.0 stabilised training across all runs tested.

Closes #55

# ✗ Meaningless
fix bug
update
wip
try this
```

### Pull Requests

- One concern per PR. A refactor and an experiment change belong in separate PRs.
- Describe **what** changed, **why**, and **how to verify** it.
- For ML changes: include metric comparisons or loss curves in the description.
- All CI checks must pass before requesting review.
- Notebooks must have all outputs cleared before being committed.

---

## 19. Code Review Checklist

Run through this before marking a PR ready for review. Fix anything that does not pass.

```
[ ] Code runs end-to-end locally on synthetic or small real data
[ ] All existing tests pass
[ ] New logic has tests covering normal cases, edge cases, and failure paths
[ ] No print() statements or debug artefacts remain
[ ] No hardcoded paths, magic numbers, or values that belong in config
[ ] Arrays and tensors are named by content, not x / data / out / tmp
[ ] All public functions and classes have docstrings with Args, Returns, Raises
[ ] Comments explain WHY — not WHAT
[ ] No dead code or commented-out blocks
[ ] Type annotations are present on all function signatures
[ ] .env.example updated if new environment variables were added
[ ] No linter errors or type errors (ruff check . && mypy src/)
[ ] Experiment results are logged in the tracker, not only in the PR
[ ] Seed is fixed, config is complete, Git hash is logged
[ ] Notebook outputs cleared before committing
[ ] Copilot suggestions have been read, understood, and cleaned up
```

---

## 20. Tooling

These tools enforce standards automatically. Configure them once — they are not optional.

| Tool | Purpose |
|---|---|
| `ruff` | Linting and formatting — replaces flake8, isort, black |
| `mypy` | Static type checking |
| `pytest` + `pytest-cov` | Test runner and coverage |
| `pre-commit` | Pre-commit hooks: lint, type-check, large-file guard |
| `pip-audit` | Dependency vulnerability scanning in CI |
| MLflow or W&B | Experiment tracking |
| DVC | Data and model versioning |

### Minimal `pyproject.toml` Config

```toml
[tool.ruff]
line-length    = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "PTH", "RUF"]

[tool.mypy]
python_version      = "3.11"
strict              = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "--cov=src --cov-report=term-missing"
```

### Minimal `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: detect-private-key
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

Code that does not pass linting or type checking is not committed. If a rule needs changing, change it through the review process — do not suppress warnings inline without a justification comment.

---

## 21. Key Principles

| Principle | Rule |
|---|---|
| Single Responsibility | One function, one class, one module — one job |
| DRY | Extract repeated logic into a shared utility. Three occurrences is the refactor threshold. |
| KISS | The simplest solution that meets requirements is the right one |
| YAGNI | Do not build for hypothetical future requirements |
| Fail Fast | Validate at every boundary. Crash loudly on bad input. |
| Reproducibility | Every result must be traceable to a seed, config, dataset, and Git commit |
| Explicitness | State every assumption — shapes, units, ranges, and types — never leave them implicit |
| Testability | If something is hard to test, the design is too tightly coupled — fix the design |
| Immutability | Do not mutate inputs. Return new objects. Make side effects deliberate and documented. |
| Explicit Dependencies | Pass config and data through function arguments. Avoid global mutable state. |
| Measure First | Profile before optimising. Every performance trade-off must be justified by data. |
| Data Governance | No sensitive data, PII, or credentials in version control — ever |

---

*This document is a living standard. When a new pattern is adopted or an old one retired, update this file in the same PR that introduces the change.*