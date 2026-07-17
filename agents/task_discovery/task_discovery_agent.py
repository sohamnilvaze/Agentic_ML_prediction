from dataclasses import dataclass
from typing import List

import pandas as pd

from agents.base_agent import BaseAgent
from agents.task_discovery.criteria import SuitabilityCriterion
from core.artifacts import DataArtifact
from core.config import REPORT_DIR
from core.constants import (
    MINIMUM_POSITIVE_SAMPLES,
    MINIMUM_SUITABILITY_SCORE,
)
from core.reporting import save_csv_report, save_json_report
from core.state import PredictionTask, WorkflowState
from core.suitability_result import SuitabilityResult
from core.utils import load_json, log_info


class FrequencyCriterion(SuitabilityCriterion):
    name = "Frequency"
    weight = 0.7

    def evaluate(self, diagnosis, diagnosis_stats, total_samples, context=None):
        max_frequency = context["max_frequency"]
        return diagnosis_stats["count"] / max_frequency


class BalanceCriterion(SuitabilityCriterion):
    name = "Balance"
    weight = 0.3

    def evaluate(self, diagnosis, diagnosis_stats, total_samples, context=None):
        positive = diagnosis_stats["count"]
        negative = total_samples - positive
        return (2 * min(positive, negative)) / (positive + negative)


@dataclass
class TaskCandidate:
    diagnosis: str
    suitability: SuitabilityResult


class TaskDiscoveryAgent(BaseAgent):
    """
    Discovers candidate prediction tasks from the master dataset.
    """

    def __init__(self):
        self.master_df = None
        self.diagnosis_distribution = None
        self.dataset_profile = None
        self.dataset_summary = None
        self.criteria = [FrequencyCriterion(), BalanceCriterion()]
        log_info("TaskDiscoveryAgent initialized.")

    def load_artifacts(self, state: WorkflowState) -> None:
        log_info("Loading preprocessing artifacts...")
        self.master_df = pd.read_csv(state.master_dataset_path)
        self.diagnosis_distribution = pd.read_csv(state.diagnosis_distribution_path)
        self.dataset_profile = load_json(state.dataset_profile_path)
        self.dataset_summary = load_json(state.dataset_summary_path)
        log_info("Artifacts loaded successfully.")

    def validate_inputs(self) -> None:
        log_info("Validating input artifacts...")
        required_columns = {"diagnosis", "count"}
        missing = required_columns - set(self.diagnosis_distribution.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        log_info("Input validation completed.")

    def generate_reasoning(
        self,
        diagnosis,
        diagnosis_stats,
        suitability_score,
        criterion_scores,
    ):
        positive = diagnosis_stats["count"]
        reasoning = [
            f"Diagnosis: {diagnosis}",
            f"Positive samples: {positive}",
            f"Frequency score: {criterion_scores['Frequency']:.3f}",
            f"Balance score: {criterion_scores['Balance']:.3f}",
            f"Overall suitability: {suitability_score:.3f}",
        ]
        if suitability_score >= 0.75:
            reasoning.append("Highly suitable for predictive modelling.")
        elif suitability_score >= 0.50:
            reasoning.append("Suitable for predictive modelling.")
        else:
            reasoning.append("Low priority candidate due to limited suitability.")
        return "\n".join(reasoning)

    def compute_suitability_score(
        self,
        diagnosis,
        diagnosis_stats,
        total_samples,
        context,
    ) -> SuitabilityResult:
        weighted_score = 0.0
        total_weight = 0.0
        criterion_scores = {}
        criterion_explanations = {}

        for criterion in self.criteria:
            score = criterion.evaluate(
                diagnosis=diagnosis,
                diagnosis_stats=diagnosis_stats,
                total_samples=total_samples,
                context=context,
            )
            criterion_scores[criterion.name] = score
            criterion_explanations[criterion.name] = ""
            weighted_score += criterion.weight * score
            total_weight += criterion.weight

        if total_weight > 0:
            weighted_score /= total_weight

        reasoning = self.generate_reasoning(
            diagnosis,
            diagnosis_stats,
            weighted_score,
            criterion_scores,
        )

        return SuitabilityResult(
            final_score=round(weighted_score, 4),
            criterion_scores=criterion_scores,
            criterion_explanations=criterion_explanations,
            reasoning=reasoning,
            metadata={
                "positive_samples": diagnosis_stats["count"],
                "negative_samples": total_samples - diagnosis_stats["count"],
            },
        )

    def analyze_candidate_tasks(self) -> List[PredictionTask]:
        total_samples = len(self.master_df)
        max_frequency = self.diagnosis_distribution["count"].max()
        context = {
            "max_frequency": max_frequency,
            "dataset_profile": self.dataset_profile,
            "dataset_summary": self.dataset_summary,
        }

        prediction_tasks = []
        for _, row in self.diagnosis_distribution.iterrows():
            diagnosis = row["diagnosis"]
            diagnosis_stats = {"count": int(row["count"])}
            suitability = self.compute_suitability_score(
                diagnosis,
                diagnosis_stats,
                total_samples,
                context,
            )
            positive = diagnosis_stats["count"]
            negative = total_samples - positive
            prevalence = positive / total_samples

            prediction_tasks.append(
                PredictionTask(
                    task_name=f"Predict {diagnosis}",
                    target_column="diagnosis",
                    target_value=diagnosis,
                    positive_samples=positive,
                    negative_samples=negative,
                    prevalence=round(prevalence, 4),
                    suitability_score=suitability.final_score,
                    criterion_scores=suitability.criterion_scores,
                    criterion_explanations=suitability.criterion_explanations,
                    reasoning=suitability.reasoning,
                    metadata=suitability.metadata,
                )
            )

        prediction_tasks.sort(key=lambda task: task.suitability_score, reverse=True)
        return prediction_tasks

    def filter_prediction_tasks(self, tasks: List[PredictionTask]) -> List[PredictionTask]:
        filtered = [
            task
            for task in tasks
            if task.suitability_score >= MINIMUM_SUITABILITY_SCORE
            and task.positive_samples >= MINIMUM_POSITIVE_SAMPLES
        ]

        if not filtered and tasks:
            return tasks[: min(3, len(tasks))]

        return filtered

    def generate_reports(self, state: WorkflowState, tasks: List[PredictionTask]):
        rows = [
            {
                "rank": rank,
                "task_name": task.task_name,
                "target_value": task.target_value,
                "suitability_score": task.suitability_score,
                "positive_samples": task.positive_samples,
                "negative_samples": task.negative_samples,
            }
            for rank, task in enumerate(tasks, start=1)
        ]

        csv_path = REPORT_DIR / "task_discovery_report.csv"
        json_path = REPORT_DIR / "task_discovery_report.json"

        save_csv_report(rows, csv_path)
        save_json_report({"tasks": [t.to_dict() for t in tasks]}, json_path)

        state.add_artifact(
            DataArtifact(
                name="Task Discovery Report",
                artifact_type="report",
                path=csv_path,
            )
        )
        state.add_artifact(
            DataArtifact(
                name="Task Discovery JSON",
                artifact_type="report",
                path=json_path,
            )
        )

    def update_workflow_state(self, state: WorkflowState, tasks: List[PredictionTask]):
        state.candidate_tasks = tasks
        state.current_stage = "TASK_DISCOVERY_COMPLETED"
        state.add_history(
            f"Task discovery completed with {len(tasks)} candidate tasks."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        log_info("Starting Task Discovery...")
        self.load_artifacts(state)
        self.validate_inputs()
        tasks = self.filter_prediction_tasks(self.analyze_candidate_tasks())
        self.generate_reports(state, tasks)
        self.update_workflow_state(state, tasks)
        log_info("Task Discovery completed.")
        return state
