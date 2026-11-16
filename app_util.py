from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional
from agents import Agent, Runner, ModelSettings, TResponseInputItem, SQLiteSession
from dotenv import load_dotenv
from pathlib import Path
import uuid
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

load_dotenv(override=True)
app = FastAPI()

# 1. Base Configuration setup
logging.basicConfig(
    level=logging.INFO,                           # Set minimum severity level to log
    format='%(asctime)s - %(levelname)s - %(message)s', # Define log layout structure
    datefmt='%Y-%m-%d %H:%M:%S'                    # Style the timestamp output
)

# 2. Instantiate a module-level logger 
logger = logging.getLogger(__name__)

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
    feedback_message: Optional[str] = Field(
        default=None,
        description="Required only when route is HUMAN_IN_LOOP_FLOW. This is the user-facing follow-up message or clarification prompt."
    )

class Router:
    async def route(self, decision: RouteDecision, websocket: WebSocket):
        if decision.route == "EXECUTE":
            return self.execute(decision)
        elif decision.route == "HUMAN_IN_LOOP_FLOW":
            return await self.human_feedback(decision, websocket)
        elif decision.route == "REJECT":
            return self.reject(decision)
        else:
            raise ValueError("Unknown route")

    def execute(self, decision: RouteDecision):
        return {"status": "EXECUTE", "reason": decision.reason}

    async def human_feedback(self, decision: RouteDecision, websocket: WebSocket):
        logger.info("Reason: %s, Confidence: %s", decision.reason, decision.confidence)
        await websocket.send_text(f"Reason: {decision.reason}")
        await websocket.send_text("Waiting for human feedback:")
        feedback = await websocket.receive_text()
        if not feedback.strip():
            decision.reason = "No feedback message found"
            return self.reject(decision)
        return {"status": "HUMAN_IN_LOOP_FLOW", "reason": decision.reason, "feedback_message": feedback}

    def reject(self, decision: RouteDecision):
        return {"status": "REJECT", "reason": decision.reason}


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

router_agent = Agent(
    name="Organization Router Agent",
    model="gpt-4o-mini",
    instructions=ROUTER_INSTRUCTIONS,
    output_type=RouteDecision,
    model_settings=ModelSettings(temperature=0.8)
)

async def route_traffic(prompt: list[TResponseInputItem], session: SQLiteSession):
    try:
        response = await Runner.run(router_agent, input=prompt, max_turns=2, session=session)
        return response.final_output
    except Exception as e:
        logger.error(f"ERROR: {e}")
    return None

def add_message(input_messages: list[TResponseInputItem], message: TResponseInputItem):
    input_messages.append(message)
    if len(input_messages) > 20:
        input_messages = input_messages[:-20]

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
    router = Router()
    session = generate_new_session()
    prev_session_id = ""
    input_messages: list[TResponseInputItem] = []
    try:
        while True:
            if prev_session_id != session.session_id:
                prev_session_id = session.session_id
                await websocket.send_text("Enter the message: ")
                user_prompt = await websocket.receive_text()
                if user_prompt.strip().lower() == "exit":
                    break
                if not user_prompt.strip():
                    continue;
                add_message(input_messages, {"role": "user", "content": user_prompt})
            result = await route_traffic(input_messages, session)
            if result is None:
                await websocket.send_text("ERROR: Router returns no output")
            try:
                decision = RouteDecision.model_validate(result)
                resp_map = await router.route(decision, websocket)
                if resp_map.get("status") == "HUMAN_IN_LOOP_FLOW":
                    if resp_map.get("feedback_message").strip().lower() == "exit":
                        break
                    add_message(input_messages, {"role": "assistant","content": resp_map.get("reason", "").strip()})
                    add_message(input_messages, {"role": "user", "content": resp_map.get("feedback_message", "").strip()})
                else:
                    await websocket.send_json(resp_map)
                    session = generate_new_session()
                    input_messages = []
            except ValidationError as e:
                logger.error("Invalid RouteDecision: %s", e)
                continue
        print("Exit the router flow...")
    except WebSocketDisconnect as wsd:
        logger.error("Socket Disconnect: %s", wsd)
    finally:
        session.close()
        await websocket.close()