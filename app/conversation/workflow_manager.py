"""
Workflow Manager

Manages operational banking workflow state transitions and WorkflowStack suspensions/resumptions.
"""

import logging
from typing import Optional

from app.conversation.models import ConversationState, WorkflowType

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages active workflows and WorkflowStack pause/resume transitions.
    """

    def push_workflow(self, state: ConversationState, new_workflow: WorkflowType) -> None:
        """
        Suspends state.active_workflow onto state.workflow_stack and sets new_workflow.
        """
        if state.active_workflow != WorkflowType.NONE and state.active_workflow != new_workflow:
            state.workflow_stack.append(state.active_workflow)
            logger.info("WorkflowManager: Suspended active workflow %s onto WorkflowStack", state.active_workflow.value)

        state.active_workflow = new_workflow
        logger.info("WorkflowManager: Active workflow set to %s", new_workflow.value)

    def pop_workflow(self, state: ConversationState) -> Optional[WorkflowType]:
        """
        Pops and resumes previously suspended workflow from state.workflow_stack.
        """
        if state.workflow_stack:
            resumed = state.workflow_stack.pop()
            state.active_workflow = resumed
            logger.info("WorkflowManager: Resumed workflow %s from WorkflowStack", resumed.value)
            return resumed

        state.active_workflow = WorkflowType.NONE
        logger.info("WorkflowManager: Workflow cleared to NONE")
        return None
