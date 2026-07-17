# End-to-End Repair Plan

This document turns the code review findings into a concrete implementation plan.
The goal is to make the project run end-to-end according to the intended business logic:

1. Build the master MIMIC dataset.
2. Discover candidate prediction tasks.
3. Send one task through human review.
4. Build a task-specific dataset.
5. Evaluate and select models.
6. Train the best model.
7. Generate explainability outputs.
8. Register the trained artifacts.

---

## 1) Bugs And Inconsistencies Blocking End-To-End Execution

### A. Workflow contract mismatches

- [`workflow/workflow_engine.py`](./workflow/workflow_engine.py) assumes every agent exposes `run(state) -> state`, but not all agents follow that contract.
- [`agents/task_discovery.py:560`](./agents/task_discovery.py#L560) returns `(state, result)` instead of only `state`.
- [`agents/dataset_builder/dataset_builder.py:39`](./agents/dataset_builder/dataset_builder.py#L39) returns a `DatasetArtifact` instead of updating and returning `WorkflowState`.
- [`agents/model_training/model_training_agent.py:64`](./agents/model_training/model_training_agent.py#L64) updates `state.training_artifact` and returns `state`, but the downstream data flow is not consistently defined across all stages.

Impact:
- The workflow engine cannot safely chain stages.
- Later agents may receive tuples or artifacts instead of `WorkflowState`.

### B. `main.py` is not wired to actual APIs

- [`main.py:27`](./main.py#L27) contains placeholder `...` code.
- [`main.py:111`](./main.py#L111) passes incomplete criteria lists.
- [`main.py:119`](./main.py#L119) constructs `DatasetBuilderAgent` with keyword names that do not match the actual constructor in [`agents/dataset_builder/dataset_builder.py:7`](./agents/dataset_builder/dataset_builder.py#L7).
- [`main.py:145`](./main.py#L145) constructs `ModelTrainingAgent`, but the surrounding profiler/evaluator APIs do not match the current implementations.
- [`main.py:187`](./main.py#L187) references `TaskDiscoveryAgent()` without importing it.
- [`main.py:189`](./main.py#L189) references `HumanReviewAgent()` even though the repo currently contains `HumanReviewer` in [`human/reviewer.py`](./human/reviewer.py).

Impact:
- The entrypoint cannot run as-is.
- The intended pipeline is not actually assembled.

### C. Duplicate legacy modules shadow the newer package structure

- [`agents/dataset_builder.py`](./agents/dataset_builder.py) duplicates the package path name `agents.dataset_builder`.
- Python resolves `agents.dataset_builder` to the flat module first, which means the intended package files under `agents/dataset_builder/` are shadowed.
- The legacy module is incomplete and contains many `NotImplementedError` stubs around [`agents/dataset_builder.py:339`](./agents/dataset_builder.py#L339).

Impact:
- Imports can land in the wrong implementation.
- The repo has two competing dataset-builder designs.

### D. Feature evaluation context field typo

- [`core/feature_evaluation_context.py:24`](./core/feature_evaluation_context.py#L24) defines `statisttics` instead of `statistics`.
- [`agents/dataset_builder/feature_evaluator.py:63`](./agents/dataset_builder/feature_evaluator.py#L63) passes `statistics=...`.

Impact:
- The dataclass constructor will fail immediately.
- Feature scoring cannot start.

### E. Model evaluation is incomplete

- [`agents/model_training/model_evaluator.py:48`](./agents/model_training/model_evaluator.py#L48) calls `self.evaluate(...)`.
- The class does not define an `evaluate()` method.
- [`agents/model_training/model_evaluator.py:110`](./agents/model_training/model_evaluator.py#L110) defines `_run_all_criteria()` but never completes the public evaluation flow.

Impact:
- Candidate models cannot be scored.
- Model selection cannot work.

### F. Model training agent calls the wrong profiler method

- [`agents/model_training/model_training_agent.py:74`](./agents/model_training/model_training_agent.py#L74) calls `self.profiler.profile(...)`.
- [`agents/model_training/dataset_profiler.py:18`](./agents/model_training/dataset_profiler.py#L18) exposes `profile_dataset(...)` instead.

Impact:
- Model training fails before candidate generation.

### G. Human review returns an undefined status

- [`human/reviewer.py:107`](./human/reviewer.py#L107) returns `AgentStatus.HUMAN_REJECTED`.
- [`core/constants.py:15`](./core/constants.py#L15) does not define `HUMAN_REJECTED`.

Impact:
- Rejection path raises an attribute error.
- Human-in-the-loop control flow is broken.

### H. Config paths are environment-specific

- [`core/config.py:16`](./core/config.py#L16) hardcodes a Colab path: `/content/drive/MyDrive/Agentic_MIMIC_POC`.

Impact:
- Local execution in this workspace will not find or write the expected files.
- The project is not portable across environments.

### I. Preprocessing and downstream state are not aligned

- The master preprocessing pipeline writes files under the configured MIMIC paths in [`core/preprocessing.py`](./core/preprocessing.py), but `main.py` loads `"data/titanic.csv"` directly.
- `WorkflowState` expects artifact paths like `master_dataset_path`, `diagnosis_distribution_path`, `dataset_profile_path`, and `dataset_summary_path` in [`core/state.py`](./core/state.py), but `main.py` only populates `raw_dataset`.

Impact:
- Task discovery cannot load the required artifacts.
- The intended business flow starts from the wrong dataset.

### J. Dataset-builder API drift and unfinished code

- [`agents/dataset_builder.py`](./agents/dataset_builder.py) contains an older end-to-end flow and many `NotImplementedError` methods.
- [`agents/dataset_builder/dataset_builder.py`](./agents/dataset_builder/dataset_builder.py) is a newer constructor-style implementation, but `main.py` does not match its signature.

Impact:
- There is no single canonical dataset-builder implementation.
- The repo cannot decide which pipeline to execute.

### K. Additional schema drift between core and agent layers

- `core/feature_selection.py` and `core/feature_selection_result.py` represent two different `FeatureSelectionResult` concepts.
- `core/dataset_profiler.py` and `agents/model_training/dataset_profiler.py` expose different profiling APIs and output shapes.
- `core/feature_evaluator.py` and `agents/dataset_builder/feature_evaluator.py` are two different evaluators with different constructor contracts.

Impact:
- The codebase has duplicate abstractions.
- Integration points are ambiguous and fragile.

---

## 2) What `main.py` Must Be Wired To

To become runnable, `main.py` needs to use one coherent set of APIs and one clear execution order.

### Required startup sequence

1. Load or generate the master dataset.
2. Populate `WorkflowState` with file paths and any initial values needed by Task Discovery.
3. Instantiate the canonical Task Discovery agent.
4. Instantiate a human review step with a valid status contract.
5. Instantiate the canonical Dataset Builder stack.
6. Instantiate the canonical Model Training stack.
7. Instantiate the Explainability pipeline.
8. Instantiate the Model Registry stack.
9. Run the workflow engine on `WorkflowState`.

### Concrete wiring fixes required in `main.py`

- Replace placeholder `...` blocks with real criteria and constructors.
- Import the actual task discovery class and the actual human review class.
- Use the dataset preprocessor output, not a sample `"titanic.csv"` demo file, if the target is the MIMIC business flow.
- Ensure `WorkflowState` contains:
  - `master_dataset_path`
  - `diagnosis_distribution_path`
  - `dataset_profile_path`
  - `dataset_summary_path`
  - `selected_prediction_task`
  - `dataset_artifact`
  - `training_artifact`
  - `explainability_artifact`
  - `registry_artifact`
- Match constructor signatures exactly:
  - `FeatureEvaluator(config=...)`
  - `DatasetBuilderAgent(feature_generator, feature_evaluator, feature_selector, dataset_constructor, dataset_validator, report_builder)`
  - `ModelTrainingAgent(profiler, model_generator, evaluator, selector, training_pipeline)`
  - `ExplainabilityAgent(pipeline=...)`
  - `ModelRegistryAgent(model_saver=...)`

### Main orchestration choice to make

We should choose one of these two patterns and apply it consistently:

1. Stateful pattern: every agent mutates `WorkflowState` and returns `WorkflowState`.
2. Artifact-return pattern: every agent returns a domain artifact and `main.py` manually stores it into `WorkflowState`.

For this repo, the stateful pattern is the better fit because:
- `WorkflowEngine` already expects it.
- Human review and registry stages already conceptually depend on shared state.
- It simplifies iteration control and future loop-back logic.

---

## 3) Phase-By-Phase Repair Plan

### Phase 1. Lock the canonical architecture

Goal:
- Stop the codebase from having two competing implementations for the same stage.

Work:
- Decide which modules are canonical:
  - Task discovery
  - Dataset builder
  - Dataset profiling
  - Feature evaluation
  - Model evaluation
  - Human review
- Mark legacy modules as deprecated or remove them.
- Eliminate namespace shadowing from `agents/dataset_builder.py`.
- Align naming across `core/*` and `agents/*`.

Exit criteria:
- Every pipeline stage has exactly one active implementation path.
- Imports resolve unambiguously.

### Phase 2. Repair shared contracts and data models

Goal:
- Make all stages speak the same language.

Work:
- Fix `FeatureEvaluationContext.statistics`.
- Review `CriterionScore`, `FeatureEvaluationResult`, `ModelEvaluationResult`, `FeatureSelectionResult`, `TrainingResult`, and `AgentResult` for compatibility.
- Ensure every result object has the fields downstream consumers actually read.
- Add `to_dict()` methods where needed for reports and JSON serialization.
- Standardize agent return values and state mutations.

Exit criteria:
- Dataclass construction succeeds across the pipeline.
- No stage requires a field that the previous stage never populates.

### Phase 3. Make preprocessing and task discovery reliable

Goal:
- Ensure the project can produce and consume the master artifacts required by Task Discovery.

Work:
- Remove the hardcoded Colab path in `core/config.py`.
- Make preprocessing output directories relative to the repo or configurable.
- Validate preprocessing outputs exist before task discovery starts.
- Ensure `TaskDiscoveryAgent` reads the actual file paths from `WorkflowState`.
- Normalize the task discovery return shape to match the engine contract.
- Fix the human review decision enum issue.

Exit criteria:
- A preprocessing run writes all required artifacts.
- Task Discovery runs without needing manual patching.
- Human review can approve or reject a task without attribute errors.

### Phase 4. Rebuild the dataset builder path

Goal:
- Convert the chosen task into a clean, validated training dataset.

Work:
- Choose the canonical dataset builder implementation and retire the other one.
- Implement or verify:
  - feature generation
  - feature evaluation
  - feature ranking/selection
  - dataset construction
  - dataset validation
  - report generation
- Make sure the feature evaluation stack uses the corrected statistics context.
- Ensure the dataset builder stores `DatasetArtifact` in `WorkflowState`.

Exit criteria:
- One selected task can produce a `DatasetArtifact`.
- Validation report and dataset report are generated.
- No `NotImplementedError` remains in the active path.

### Phase 5. Repair model evaluation and training

Goal:
- Select candidate models, train them, and choose the best performer.

Work:
- Implement `ModelEvaluator.evaluate()`.
- Confirm `ModelScoringPolicy` receives the right criterion names and score objects.
- Fix `ModelTrainingAgent` to call the correct profiler method.
- Verify `DatasetProfiler` output matches what model criteria expect.
- Confirm `TrainingPipeline` can split data, train, score, cross-validate, and select the best model.
- Review `TrainingResult` field order and constructor usage.

Exit criteria:
- Candidate models are generated and evaluated successfully.
- At least one model is trained end-to-end.
- `TrainingArtifact` contains the selected best model.

### Phase 6. Finish explainability and registry

Goal:
- Produce post-training artifacts and persist the best model.

Work:
- Verify `ExplainabilityPipeline` can handle both tree-based importances and linear coefficients.
- Make it robust when no feature importance is available.
- Ensure feature names align with training preprocessing.
- Verify `ModelSaver` persists both model and metadata.
- Ensure `ModelRegistryAgent` writes `RegistryArtifact` into `WorkflowState`.

Exit criteria:
- Explainability output is created for trained models.
- Registered model files and metadata exist on disk.

### Phase 7. Rebuild the top-level orchestration

Goal:
- Make `main.py` a true entrypoint rather than a sketch.

Work:
- Replace demo-only loading with the real preprocessing/bootstrap flow.
- Instantiate all canonical agents.
- Run the workflow in the correct order.
- Add clear logging and failure handling.
- Ensure the final state exposes all final artifacts.

Exit criteria:
- Running `main.py` executes the full business flow without manual edits.

### Phase 8. Add verification and regression protection

Goal:
- Keep the system working after the repair.

Work:
- Add unit tests for:
  - feature statistics
  - feature criteria
  - model scoring policy
  - dataset validation
  - workflow engine state transitions
- Add an integration test for the full pipeline on a small fixture dataset.
- Add a smoke test that verifies the bootstrap flow can reach registry output.

Exit criteria:
- The project has at least one end-to-end smoke test.
- Future regressions in contracts or orchestration are caught early.

---

## 4) Recommended Repair Order

1. Fix shared data contracts and the `statistics` typo.
2. Remove the `agents/dataset_builder` namespace collision.
3. Make `WorkflowEngine` and all agents use one return contract.
4. Repair `TaskDiscoveryAgent` and `HumanReviewer`.
5. Repair the dataset-builder pipeline.
6. Implement `ModelEvaluator.evaluate()` and fix model-training calls.
7. Align config paths and bootstrap the real preprocessing flow.
8. Wire `main.py` to the canonical API set.
9. Add tests and smoke validation.

---

## 5) Success Definition

The project is done when this sequence works without manual code edits:

1. Preprocess raw MIMIC tables into a master dataset.
2. Discover candidate prediction tasks.
3. Approve one task via human review.
4. Build and validate the task-specific dataset.
5. Score and select candidate models.
6. Train the selected model(s).
7. Generate explainability output.
8. Register the final model artifact.

That is the bar for “end-to-end runnable as intended.”
