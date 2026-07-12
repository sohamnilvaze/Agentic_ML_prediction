import os
import json
import pandas as pd

from agents.base_agent import BaseAgent
from core.agent_result import AgentResult
from core.candidate_feature import CandidateFeature, ExclusionType
from core.candidate_feature import ExclusionType
from core.dataset_profiler import DatasetProfiler
from core.state import WorkflowState


def _summarize_candidate_features(
    self
):

    total = len(
        self.candidate_features
    )

    candidate = sum(

        f.candidate

        for f in self.candidate_features

    )

    excluded = total - candidate

    hard = sum(

        f.exclusion_type == ExclusionType.HARD

        for f in self.candidate_features

    )

    soft = sum(

        f.exclusion_type == ExclusionType.SOFT

        for f in self.candidate_features

    )

    self.discovery_summary = {

        "total_features": total,

        "candidate_features": candidate,

        "excluded_features": excluded,

        "hard_exclusions": hard,

        "soft_exclusions": soft

    }

def _save_candidate_features(
    self
):

    os.makedirs(
        "artifacts/dataset_builder",
        exist_ok=True
    )

    data = [

        feature.to_dict()

        for feature

        in self.candidate_features

    ]

    with open(

        "artifacts/dataset_builder/candidate_features.json",

        "w"

    ) as f:

        json.dump(

            data,

            f,

            indent=4

        )


class DatasetBuilderAgent(BaseAgent):

    """
    Constructs a machine-learning-ready dataset
    for the selected prediction task.
    """

    def __init__(self):

        self.master_df = None

        self.dataset_profile = None

        self.candidate_features = []

        self.feature_evaluations = []

        self.selected_features = []

        self.discovery_summary = {}

    # ------------------------------------------------------

    def load_artifacts(
        self,
        state: WorkflowState
    ):

        """
        Loads required artifacts from WorkflowState.
        """

        self.master_df = pd.read_csv(
            state.master_dataset_path
        )

    # ------------------------------------------------------

    def validate_inputs(
        self,
        state: WorkflowState
    ):

        """
        Validate required workflow inputs.
        """

        if self.master_df is None:

            raise ValueError(
                "Master dataset not loaded."
            )

        if state.selected_prediction_task is None:

            raise ValueError(
                "No prediction task selected."
            )

    # ------------------------------------------------------

    def profile_dataset(
        self,
        state
    ):

        print("=" * 80)
        print("Profiling Dataset")
        print("=" * 80)

        profiler = DatasetProfiler()

        self.dataset_profile = profiler.build_profile(

            self.master_df,

            target_column=(
                state.selected_prediction_task.target_column
            )

        )

        state.dataset_profile = self.dataset_profile

        print(

            f"Samples : {self.dataset_profile.n_samples}"

        )

        print(

            f"Features : {self.dataset_profile.n_features}"

        )

        print(

            f"Warnings : {len(self.dataset_profile.warnings)}"

        )

        print(

            "Dataset profiling completed."

        )

        os.makedirs(
            "artifacts/dataset_builder",
            exist_ok=True
        )

        with open(
            "artifacts/dataset_builder/dataset_profile.json",
            "w"
        ) as f:

            json.dump(
                self.dataset_profile.to_dict(),
                f,
                indent=4
            )
    # ------------------------------------------------------

    def build_candidate_objects(
        self
    ):

        profile = self.dataset_profile

        self.candidate_features = []

        for column in profile.feature_names:

            feature = CandidateFeature(

                feature_name=column,

                dtype=profile.feature_types[column],

                missing_percentage=profile.missing_percentages[column],

                unique_count=profile.unique_counts[column],

                is_numeric=(
                    column in profile.numeric_columns
                ),

                is_metadata=(
                    column in profile.metadata_columns
                ),

                is_constant=(
                    column in profile.constant_columns
                ),

                is_high_cardinality=(
                    column in profile.high_cardinality_columns
                )

            )

            self.candidate_features.append(
                feature
            )
    
    def apply_hard_exclusion_rules(
        self,
        target_column
    ):

        for feature in self.candidate_features:

            if feature.feature_name == target_column:

                feature.candidate = False

                feature.exclusion_type = (
                    ExclusionType.HARD
                )

                feature.exclusion_reason = (
                    "Target column."
                )

                continue

            if feature.is_metadata:

                feature.candidate = False

                feature.exclusion_type = (
                    ExclusionType.HARD
                )

                feature.exclusion_reason = (
                    "Metadata column."
                )

                continue

            if feature.is_constant:

                feature.candidate = False

                feature.exclusion_type = (
                    ExclusionType.HARD
                )

                feature.exclusion_reason = (
                    "Constant column."
                )

    def discover_candidate_features(
        self,
        state
    ):

        print("=" * 80)
        print("Discovering Candidate Features")
        print("=" * 80)

        self.build_candidate_objects()

        self.apply_hard_exclusion_rules(

            state.selected_prediction_task.target_column

        )

        state.candidate_features = self.candidate_features

        self._summarize_candidate_features()

        self._save_candidate_features()

        print(...)
    # ------------------------------------------------------

    def evaluate_candidate_features(
        self,
        state: WorkflowState
    ):

        raise NotImplementedError

    # ------------------------------------------------------

    def select_features(
        self,
        state: WorkflowState
    ):

        raise NotImplementedError

    # ------------------------------------------------------

    def construct_dataset(
        self,
        state: WorkflowState
    ):

        raise NotImplementedError

    # ------------------------------------------------------

    def validate_dataset(
        self,
        state: WorkflowState
    ):

        raise NotImplementedError

    # ------------------------------------------------------

    def generate_reports(
        self,
        state: WorkflowState
    ):

        raise NotImplementedError

    # ------------------------------------------------------

    def update_workflow_state(
        self,
        state: WorkflowState
    ):

        raise NotImplementedError

    # ------------------------------------------------------

    def create_agent_result(self):

        raise NotImplementedError

    # ------------------------------------------------------

    def run(
        self,
        state: WorkflowState
    ):

        self.load_artifacts(state)

        self.validate_inputs(state)

        self.profile_dataset(state)

        self.discover_candidate_features(state)

        self.evaluate_candidate_features(state)

        self.select_features(state)

        self.construct_dataset(state)

        self.validate_dataset(state)

        self.generate_reports(state)

        self.update_workflow_state(state)

        return (
            state,
            self.create_agent_result()
        )