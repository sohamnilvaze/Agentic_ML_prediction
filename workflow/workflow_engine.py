from core.state import WorkflowState


class WorkflowEngine:
    """
    Executes all workflow agents sequentially.

    Each agent receives the current WorkflowState
    and returns the updated WorkflowState.
    """

    # --------------------------------------------------

    def __init__(

        self,

        agents

    ):

        self.agents = agents

    # ==================================================

    def run_until(

        self,

        state,

        agent_type

    ):

        for agent in self.agents:

            state = agent.run(

                state

            )

            if isinstance(

                agent,

                agent_type

            ):

                break

        return state

    def run(

        self,

        initial_state: WorkflowState

    ) -> WorkflowState:

        state = initial_state

        for agent in self.agents:

            state = agent.run(

                state

            )

        return state