import logging
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Protocol, Optional
from agents.exceptions import AgentsException

# 1. Base Configuration setup
logging.basicConfig(
    level=logging.INFO,                           # Set minimum severity level to log
    format='%(asctime)s - %(levelname)s - %(message)s', # Define log layout structure
    datefmt='%Y-%m-%d %H:%M:%S'                    # Style the timestamp output
)

# 2. Instantiate a module-level logger 
logger = logging.getLogger(__name__)

class ValidationResponse(BaseModel):
    message: str = Field(description="Validation message indicates what happen during validation")
    status: int = Field(description="Status to indicates the type of validation error")

class AgentNoResponseError(AgentsException):
    """Raised when the agent completes execution but returns no content."""
    pass

class Capability(BaseModel):
    name: str = Field(description="Name of the capability")
    description: str = Field(description="Description of the capability")

class RouteDecision(BaseModel):
    route: Literal["EXECUTE", "HUMAN_IN_LOOP_FLOW", "REJECT"]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.00 to 1.00."
    )
    reason: str = Field(
        min_length=1,
        description="Reason for selecting this route."
    )
    feedback_message: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_route_rules(self):
        if self.route == "EXECUTE" and self.feedback_message is not None:
            raise ValueError("EXECUTE route must not include feedback_message.")
        if self.route == "HUMAN_IN_LOOP_FLOW" and not self.feedback_message:
            raise ValueError("HUMAN_IN_LOOP_FLOW route requires feedback_message.")
        if self.route == "REJECT" and self.feedback_message:
            raise ValueError("REJECT route should not include feedback_message.")
        return self

class ExternalAgentResponse(BaseModel):
    message: str
    status: Literal["done", "pending", "failure"]

class Route(Protocol):
    def execute(self, decision: RouteDecision):
        ...

    def human_feedback(self, decision: RouteDecision, feedback: str):
        ...

    def reject(self, decision: RouteDecision):
        ...