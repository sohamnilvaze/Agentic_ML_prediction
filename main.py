import pandas as pd

from agents.dataset_builder.dataset_builder import DatasetBuilderAgent
from agents.dataset_builder.dataset_report_builder import DatasetReportBuilder
from agents.dataset_builder.dataset_validator import DatasetValidator
from core.dataset_construction import DatasetConstructor
from core.feature_selector import FeatureSelector
from workflow.workflow_engine import WorkflowEngine

from core.state import WorkflowState

# =====================================================
# Task Discovery
# =====================================================

# =====================================================
# Human Review
# =====================================================


# =====================================================
# Dataset Builder
# =====================================================

# Import your feature criteria

...

# =====================================================
# Model Training
# =====================================================

from agents.model_training.model_training_agent import (
    ModelTrainingAgent
)

from agents.model_training.dataset_profiler import (
    DatasetProfiler
)

from agents.model_training.candidate_model_generator import (
    CandidateModelGenerator
)

from agents.model_training.model_evaluator import (
    ModelEvaluator
)

from agents.model_training.model_selector import (
    ModelSelector
)

from agents.model_training.training_pipeline import (
    TrainingPipeline
)

# =====================================================
# Explainability
# =====================================================

from agents.explainability.explainability_agent import (
    ExplainabilityAgent
)

from agents.explainability.explainability_pipeline import (
    ExplainabilityPipeline
)

# =====================================================
# Registry
# =====================================================

from agents.model_registry.model_registry_agent import (
    ModelRegistryAgent
)

from agents.model_registry.model_saver import (
    ModelSaver
)

# =====================================================
# Load Dataset
# =====================================================

raw_dataframe = pd.read_csv(

    "data/titanic.csv"

)

# =====================================================
# Initial Workflow State
# =====================================================

state = WorkflowState(

    raw_dataset=raw_dataframe

)

# =====================================================
# Construct Dataset Builder
# =====================================================

from agents.dataset_builder.feature_evaluator import FeatureEvaluator

feature_evaluator = FeatureEvaluator(

    criteria=[

        ...

    ]

)

dataset_builder_agent = DatasetBuilderAgent(

    evaluator=feature_evaluator,

    selector=FeatureSelector(),

    constructor=DatasetConstructor(),

    validator=DatasetValidator(),

    report_builder=DatasetReportBuilder()

)

# =====================================================
# Construct Model Training
# =====================================================

model_evaluator = ModelEvaluator(

    criteria=[

        ...

    ]

)

model_training_agent = ModelTrainingAgent(

    profiler=DatasetProfiler(),

    model_generator=CandidateModelGenerator(),

    evaluator=model_evaluator,

    selector=ModelSelector(),

    training_pipeline=TrainingPipeline()

)

# =====================================================
# Construct Explainability
# =====================================================

explainability_agent = ExplainabilityAgent(

    pipeline=ExplainabilityPipeline()

)

# =====================================================
# Construct Registry
# =====================================================

registry_agent = ModelRegistryAgent(

    model_saver=ModelSaver()

)

# =====================================================
# Workflow Engine
# =====================================================

engine = WorkflowEngine(

    [

        TaskDiscoveryAgent(),

        HumanReviewAgent(),

        dataset_builder_agent,

        model_training_agent,

        explainability_agent,

        registry_agent

    ]

)

# =====================================================
# Execute
# =====================================================

final_state = engine.run(

    state

)

print(

    "Pipeline completed successfully."

)