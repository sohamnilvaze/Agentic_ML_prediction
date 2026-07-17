from core.agent_result import AgentResult
from core.constants import AgentStatus


class HumanReviewer:

    """
    Handles all Human-In-The-Loop
    interactions.
    """

    def review_prediction_tasks(
        self,
        state
    ):

        print("\n" + "=" * 80)
        print("Candidate Prediction Tasks")
        print("=" * 80)

        for idx, task in enumerate(
            state.candidate_tasks,
            start=1
        ):

            print(f"\n[{idx}] {task.target_value}")

            print(
                f"Suitability : "
                f"{task.suitability_score:.3f}"
            )

            print(
                f"Positive Samples : "
                f"{task.positive_samples}"
            )

            print(
                task.reasoning
            )

            print("-" * 60)

        while True:

            try:

                choice = int(
                    input(
                        "\nSelect task number : "
                    )
                )

                if (
                    1 <= choice <=
                    len(state.candidate_tasks)
                ):

                    break

            except ValueError:

                pass

            print(
                "Invalid selection."
            )

        comments = input(

            "\nComments (optional): "

        )

        approve = input(

            "\nApprove? (Y/N): "

        ).strip().upper()

        state.human_feedback = comments

        if approve == "Y":

            state.selected_prediction_task = (

                state.candidate_tasks[
                    choice - 1
                ]

            )

        return AgentResult(
            status=AgentStatus.SUCCESS,
            reasoning="Prediction task approved by reviewer.",
            next_stage="Dataset Builder",
            recommendations=[],
        )

        return AgentResult(
            status=AgentStatus.FAILURE,
            reasoning="Reviewer rejected all candidate tasks.",
            next_stage="Task Discovery",
            recommendations=["Re-run Task Discovery."],
        )

    def run(self, state):
        """
        Non-interactive workflow hook.

        For unattended runs, approve the top-ranked candidate task when
        available and keep the manual review path available through
        review_prediction_tasks().
        """

        if not state.candidate_tasks:
            raise ValueError("No candidate tasks available for review.")

        state.selected_prediction_task = state.candidate_tasks[0]
        state.human_feedback = "Auto-approved top-ranked candidate task."
        state.current_stage = "HUMAN_REVIEW_COMPLETED"
        state.add_history(
            f"Human review auto-selected task: {state.selected_prediction_task.task_name}"
        )
        return state


HumanReviewAgent = HumanReviewer
