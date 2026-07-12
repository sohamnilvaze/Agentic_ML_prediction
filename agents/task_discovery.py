

from attr import dataclass
import pandas as pd
from typing import List

from agents.base_agent import BaseAgent
from agents.task_discovery.criteria import SuitabilityCriterion
from core import state
from core.agent_result import AgentResult
from core.artifacts import DataArtifact
from core.config import REPORT_DIR
from core.constants import MINIMUM_POSITIVE_SAMPLES, MINIMUM_SUITABILITY_SCORE, AgentStatus
from core.reporting import save_csv_report, save_json_report
from core.state import PredictionTask, WorkflowState
from core.suitability_result import SuitabilityResult
from core.utils import load_json, log_info

class FrequencyCriterion(
    SuitabilityCriterion
):

    name = "Frequency"

    weight = 0.7

    def evaluate(
        self,
        diagnosis,
        diagnosis_stats,
        total_samples,
        context=None
    ):

        max_frequency = context["max_frequency"]

        return (
            diagnosis_stats["count"]
            /
            max_frequency
        )


class BalanceCriterion(
    SuitabilityCriterion
):

    name = "Balance"

    weight = 0.3

    def evaluate(
        self,
        diagnosis,
        diagnosis_stats,
        total_samples,
        context=None
    ):

        positive = diagnosis_stats["count"]

        negative = total_samples - positive

        score = (
            2 *
            min(
                positive,
                negative
            )
        ) / (
            positive + negative
        )

        return score

@dataclass
class TaskCandidate:
    diagnosis: str
    suitability: SuitabilityResult


class TaskDiscoveryAgent(BaseAgent):

    """
    Discovers candidate prediction tasks
    from the master dataset.
    """

    def __init__(self):

        self.master_df = None

        self.diagnosis_distribution = None

        self.dataset_profile = None

        self.dataset_summary = None

        self.criteria = [

            FrequencyCriterion(),

            BalanceCriterion()

        ]

        log_info(
            "TaskDiscoveryAgent initialized."
        )
    
    @property
    def capabilities(self):

        return [

            "task_discovery",

            "frequency_analysis",

            "candidate_generation"

        ]
    
    def load_artifacts(
        self,
        state: WorkflowState
    ) -> None:

        log_info(
            "Loading preprocessing artifacts..."
        )

        self.master_df = pd.read_csv(
            state.master_dataset_path
        )

        self.diagnosis_distribution = pd.read_csv(
            state.diagnosis_distribution_path
        )

        self.dataset_profile = load_json(
            state.dataset_profile_path
        )

        self.dataset_summary = load_json(
            state.dataset_summary_path
        )

        log_info(
            "Artifacts loaded successfully."
        )
    
    def validate_inputs(self):

        log_info(
            "Validating input artifacts..."
        )

        required_columns = {

            "diagnosis",

            "count"

        }

        missing = required_columns - set(
            self.diagnosis_distribution.columns
        )

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        log_info(
            "Input validation completed."
        )
    

    def generate_reasoning(
        self,
        diagnosis,
        diagnosis_stats,
        suitability_score,
        criterion_scores
    ):

        positive = diagnosis_stats["count"]

        frequency = criterion_scores["Frequency"]

        balance = criterion_scores["Balance"]

        reasoning = []

        reasoning.append(
            f"Diagnosis: {diagnosis}"
        )

        reasoning.append(
            f"Positive samples: {positive}"
        )

        reasoning.append(
            f"Frequency score: {frequency:.3f}"
        )

        reasoning.append(
            f"Balance score: {balance:.3f}"
        )

        reasoning.append(
            f"Overall suitability: {suitability_score:.3f}"
        )

        if suitability_score >= 0.75:

            recommendation = (
                "Highly suitable for predictive modelling."
            )

        elif suitability_score >= 0.50:

            recommendation = (
                "Suitable for predictive modelling."
            )

        else:

            recommendation = (
                "Low priority candidate due to limited suitability."
            )

        reasoning.append(
            recommendation
        )

        return "\n".join(reasoning)


    def compute_suitability_score(
        self,
        diagnosis,
        diagnosis_stats,
        total_samples,
        context
    ) -> SuitabilityResult:

        weighted_score = 0.0

        total_weight = 0.0

        criterion_scores = {}

        criterion_explanations = {}

        # --------------------------------------------------
        # Evaluate every registered criterion
        # --------------------------------------------------

        for criterion in self.criteria:

            result = criterion.evaluate(
                diagnosis=diagnosis,
                diagnosis_stats=diagnosis_stats,
                total_samples=total_samples,
                context=context
            )

            score = result["score"]

            explanation = result["explanation"]

            criterion_scores[criterion.name] = score

            criterion_explanations[criterion.name] = explanation

            weighted_score += criterion.weight * score

            total_weight += criterion.weight

        # --------------------------------------------------
        # Normalize score
        # --------------------------------------------------

        if total_weight > 0:

            weighted_score /= total_weight

        # --------------------------------------------------
        # Build reasoning
        # --------------------------------------------------

        reasoning = self.generate_reasoning(
            diagnosis,
            diagnosis_stats,
            weighted_score,
            criterion_scores
        )

        return SuitabilityResult(

            final_score=round(weighted_score, 4),

            criterion_scores=criterion_scores,

            criterion_explanations=criterion_explanations,

            reasoning=reasoning,

            metadata={
                "positive_samples": diagnosis_stats["count"],
                "negative_samples":
                    total_samples - diagnosis_stats["count"]
            }
        )

    def analyze_candidate_tasks(
            self
        ) -> List[PredictionTask]:

            total_samples = len(self.master_df)

            max_frequency = (
                self.diagnosis_distribution["count"]
                .max()
            )

            context = {

                "max_frequency": max_frequency,

                "dataset_profile": self.dataset_profile,

                "dataset_summary": self.dataset_summary

            }

            prediction_tasks = []

            for _, row in self.diagnosis_distribution.iterrows():

                diagnosis = row["diagnosis"]

                diagnosis_stats = {

                    "count": row["count"]

                }

                suitability = self.compute_suitability_score(

                    diagnosis,

                    diagnosis_stats,

                    total_samples,

                    context

                )

                positive = diagnosis_stats["count"]

                negative = total_samples - positive

                prevalence = positive / total_samples

                task = PredictionTask(

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

                    metadata=suitability.metadata

                )

                prediction_tasks.append(task)

            prediction_tasks.sort(

                key=lambda task: task.suitability_score,

                reverse=True

            )

            return prediction_tasks

    def filter_prediction_tasks(
        self,
        tasks: List[PredictionTask]
    ) -> List[PredictionTask]:

        filtered = []

        for task in tasks:

            if (
                task.suitability_score >= MINIMUM_SUITABILITY_SCORE
                and
                task.positive_samples >= MINIMUM_POSITIVE_SAMPLES
            ):

                filtered.append(task)

        return filtered

    def generate_reports(
        self,
        state,
        tasks
    ):

        rows = []

        for rank, task in enumerate(tasks, start=1):

            rows.append({

                "rank": rank,

                "task_name": task.task_name,

                "target_value": task.target_value,

                "suitability_score": task.suitability_score,

                "positive_samples": task.positive_samples,

                "negative_samples": task.negative_samples

            })

        csv_path = (
            REPORT_DIR /
            "task_discovery_report.csv"
        )

        json_path = (
            REPORT_DIR /
            "task_discovery_report.json"
        )

        save_csv_report(
            rows,
            csv_path
        )

        save_json_report(

            {

                "tasks":[
                    t.to_dict()
                    for t in tasks
                ]

            },

            json_path

        )

        state.add_artifact(

            DataArtifact(

                name="Task Discovery Report",

                artifact_type="report",

                path=csv_path

            )

        )

        state.add_artifact(

            DataArtifact(

                name="Task Discovery JSON",

                artifact_type="report",

                path=json_path

            )

        )
    
    def update_workflow_state(
        self,
        state,
        tasks
    ):

        state.candidate_tasks = tasks

        state.current_stage = "Task Discovery Completed"

        state.add_history(

            f"{len(tasks)} prediction tasks discovered."

        )
    


    def create_agent_result(
        self,
        tasks
    ):

        return AgentResult(

            status=AgentStatus.SUCCESS,

            reasoning=(

                f"Task Discovery Agent identified "

                f"{len(tasks)} suitable prediction tasks."

            ),

            next_stage="Human Review",

            recommendations=[

                "Review discovered prediction tasks.",

                "Select one task to continue."

            ]

        )

    
    def run(
        self,
        state
        ):

        log_info(
            "Starting Task Discovery..."
        )

        self.load_artifacts(state)

        self.validate_inputs()

        tasks = self.analyze_candidate_tasks()

        tasks = self.filter_prediction_tasks(
            tasks
        )

        self.generate_reports(
            state,
            tasks
        )

        self.update_workflow_state(
            state,
            tasks
        )

        result = self.create_agent_result(
            tasks
        )

        log_info(
            "Task Discovery completed."
        )

        return state, result
