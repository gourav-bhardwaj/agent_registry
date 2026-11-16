import logging
import os

from agents import Agent, ModelSettings, Runner, set_tracing_export_api_key, SQLiteSession
from agents.extensions.models.any_llm_model import AnyLLMModel
from dotenv import load_dotenv
from pydantic import ValidationError
from models import Capability, RouteDecision, AgentNoResponseError, Route, ValidationResponse
from typing import Any, Protocol
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
import uuid

load_dotenv(override=True)
app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class RouterPolicy:
    ROUTER_INSTRUCTIONS = """
    You are an Organization Request Router Agent.
    Your ONLY responsibility is to classify the user's request into exactly
    one of these routes:
    1. EXECUTE
    2. HUMAN_IN_LOOP_FLOW
    3. REJECT

    Do not execute the request.
    Do not create tasks.
    Do not call domain agents.
    Do not invent missing information.

    EXECUTE:
    Select when the user's intent is clear, the request is within scope,
    all critical information is available, there is no unresolved ambiguity,
    and no human approval is required before execution.

    HUMAN_IN_LOOP_FLOW:
    Select when the request can potentially be handled but additional
    information, clarification, or human approval is required.

    REJECT:
    Select when the request is outside the supported scope, prohibited,
    or should not be handled by this system.

    DECISION PRIORITY:
    1. Understand the user's intent.
    If intent cannot be determined -> HUMAN_IN_LOOP_FLOW.

    2. Check whether the request is supported.
    If clearly outside supported capabilities -> REJECT.

    3. Check authorization and policy.
    If prohibited -> REJECT.
    If approval is required -> HUMAN_IN_LOOP_FLOW.

    4. Check required information.
    If critical information is missing -> HUMAN_IN_LOOP_FLOW.

    5. Check ambiguity.
    If multiple reasonable interpretations could result in different
    business actions -> HUMAN_IN_LOOP_FLOW.

    6. If everything required is available and the operation can proceed: -> EXECUTE.

    Never guess critical business information.
    Critical information can include:
    employee name, employee ID, email, department, role, manager,
    destination, source, date, amount, permission, approval, or resource ID.
    Use the entire relevant conversation context when determining whether
    information has already been provided.

    EXECUTE means:
    "I know what the user wants AND I have enough information to proceed."

    HUMAN_IN_LOOP_FLOW means:
    "I understand what the user wants BUT I need information,
    clarification, or approval before proceeding."

    REJECT means:
    "I understand what the user wants BUT this system should not perform it."

    For EXECUTE:
    feedback_message must be null.

    For HUMAN_IN_LOOP_FLOW:
    feedback_message must be provided and must clearly tell the user
    what information or approval is required.

    For REJECT:
    feedback_message should normally be null.

    The confidence score represents confidence in the ROUTING DECISION,
    not confidence that the requested operation will succeed.

    The reason must explain why the route was selected.

    Return ONLY the RouteDecision structured output.
    """

    @staticmethod
    def system_prompt():
        return RouterPolicy.ROUTER_INSTRUCTIONS

    @staticmethod
    def build_prompt(agent_name: str, agent_desc: str, capabilities: list, user_request: str) -> str:
        cap_text = "\n".join(f"- {c.name}: {c.description}" for c in capabilities)
        return f"""
        Determine the appropriate route for the following agent request.
        Agent: {agent_name}
        Description: {agent_desc}
        Capabilities:
        {cap_text}

        User request: {user_request}
        """

class AgentBuilder:
    @staticmethod
    def _build_args(name: str, model: str, system_prompt: str, model_output: Any, model_settings: ModelSettings, using_openai_model: bool = True):
        try:
            if using_openai_model:
                agent = Agent(
                    name=name,
                    model=model,
                    instructions=system_prompt,
                    output_type=model_output,
                    model_settings=model_settings,
                )
                logger.info("%s Agent initialized with OpenAI model: %s", name, model)
                return agent

            # If not using OPENAI Configuration within the application. 
            tracing_api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")
            api_key = os.getenv("API_KEY")

            if not base_url or not api_key:
                raise ValueError("BASE_URL and API_KEY must be configured for non-OpenAI models.")

            set_tracing_export_api_key(tracing_api_key)
            non_openai_model = AnyLLMModel(
                model=model,
                base_url=base_url,
                api_key=api_key,
            )

            agent = Agent(
                name=name,
                model=non_openai_model,
                instructions=system_prompt,
                output_type=model_output,
                model_settings=model_settings,
            )
            logger.info("%s Agent initialized with Non-OpenAI Model: %s", name, model)
            return agent
        except Exception as exc:
            logger.exception("%s Agent init error: %s", name, exc)
            raise

class RouterClassifier:
    def __init__(self, model: str, session: SQLiteSession, using_openai_model: bool = True):
        self.router_agent = AgentBuilder._build_args("Organization Router Agent", 
                                                     model, RouterPolicy.system_prompt(), 
                                                     RouteDecision, ModelSettings(temperature=0.1), 
                                                     using_openai_model)
        self.session = session
    
    async def classify(self, prompt: str) -> RouteDecision:
        try:
            result = await Runner.run(self.agent, input=prompt, max_turns=2, session=self.session)
            return result.final_output
        except Exception as e:
            logger.error(f"ERROR: {e}")


class RouteDecisionValidator:
    @staticmethod
    def validate(value: Any) -> RouteDecision:
        try:
            if value is None:
                raise ValueError("Router returned no decision.")
            return RouteDecision.model_validate(value)
        except (ValidationError, TypeError) as ve:
            logger.error("ValidationError: %s", ve)
            raise

class HumanFeedbackHandler:
    async def request(self, decision: RouteDecision, classifier: RouterClassifier, websocket: WebSocket):
        await websocket.send_text(f"Reason: {decision.reason}")
        await websocket.send_text("Waiting for human feedback:")
        feedback = await websocket.receive_text()

        if not feedback or not feedback.strip():
            return {
                "status": "REJECT",
                "reason": "No feedback message found"
            }

        return classifier.classify(prompt=f"""
        Reasoning For Feedback: {decision.reason}
        Feedback: {feedback}
        """)

class RouteExecutor:
    def execute(self): 
        logger.info("Execute to perform the operation")

    def reject(self):
        logger.info("Reject the operation")

class RouteOrchestrator:
    def __init__(self, classifier: RouterClassifier, validator: RouteDecisionValidator):
        self.classifier = classifier
        self.validator = validator
        self.executor = RouteExecutor()
        self.human_handler = HumanFeedbackHandler()

    async def handle(self, user_request: str, websocket: WebSocket, agent_name: str, agent_desc: str, capabilities: list):
        prompt = RouterPolicy.build_prompt(agent_name, agent_desc, capabilities, user_request)
        decision = await self.classifier.classify(prompt)
        decision = self.validator.validate(decision)

        if decision.route == "EXECUTE":
            return self.executor.execute()

        if decision.route == "HUMAN_IN_LOOP_FLOW":
            return await self.human_handler.request(decision, self.classifier, websocket)

        if decision.route == "REJECT":
            return self.executor.reject()

        raise ValueError(f"Unknown route: {decision.route}")


DB_PATH = Path("data/conversations.db")

def generate_new_session() -> SQLiteSession:
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    session_id = "workflow_" + uuid.uuid4().hex
    return SQLiteSession(
        session_id=session_id,
        db_path=str(DB_PATH),
    )

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        session = generate_new_session()
        classifier = RouterClassifier(model="gpt-4o-mini", session=session, using_openai_model=True)
        validator = RouteDecisionValidator()
        orchestrator = RouteOrchestrator(classifier=classifier, validator=validator)
        user_query = input("User Query: ")
        await orchestrator.handle(user_request=user_query, websocket=websocket, 
                            agent_name="employee_agent", 
                            agent_desc="Specialized agent responsible for managing employee lifecycle and employee-related operations, including retrieving employee information, onboarding new employees, deboarding existing employees, and updating employee details. Operations that modify or remove employee data require human approval before execution.", 
                            capabilities=[
                                "Employee management",
                                "Employee information retrieval",
                                "Employee onboarding",
                                "Employee deboarding",
                                "Employee profile management",
                                "Employee detail updates"
                            ])
        print("Exit the router flow...")
    except WebSocketDisconnect as wsd:
        logger.error("Socket Disconnect: %s", wsd)
    finally:
        await websocket.close()
        session.close()

# class RouteDecisionValidator:
#     @staticmethod
#     def validate(response: RouteDecision | dict) -> ValidationResponse:
#         try:
#             if response is None:
#                 raise AgentNoResponseError("Router agent failed to return a valid response content.")

#             RouteDecision.model_validate(response)
#             return ValidationResponse(message="RouteDecision object is valid", status=200)

#         except (ValidationError, TypeError) as ve:
#             logger.error("ValidationError: %s", ve)
#             return ValidationResponse(message=f"ValidationError: {ve}", status=400)

#         except AgentNoResponseError as ase:
#             logger.error("AgentNoResponseError: %s", ase)
#             return ValidationResponse(message=f"AgentNoResponseError: {ase}", status=500)

# class RouterOrchestrator:
#     def __init__(self, route: Route):
#         self.route = route
        
#     def route(self, decision: RouteDecision):
#         if decision.route == "EXECUTE":
#             return self.route.execute(decision)
#         elif decision.route == "HUMAN_IN_LOOP_FLOW":
#             return self.route.human_feedback(decision)
#         elif decision.route == "REJECT":
#             return self.route.reject(decision)
#         else:
#             raise ValueError("Unknown route")