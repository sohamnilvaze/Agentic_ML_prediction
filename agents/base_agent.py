from abc import ABC, abstractmethod

from core.state import WorkflowState


class BaseAgent(ABC):
    """
    Base class for all workflow agents.

    Every agent receives the current workflow state,
    performs one stage of the workflow, and returns
    the updated workflow state.
    """

    @abstractmethod
    def run(
        self,
        state: WorkflowState
    ) -> WorkflowState:
        """
        Execute this workflow stage.
        """
        pass