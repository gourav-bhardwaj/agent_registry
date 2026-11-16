from contextlib import asynccontextmanager

from fastapi import FastAPI
from agents import Agent, Runner, ModelSettings
from pydantic import BaseModel
from typing import Literal, Any
import uvicorn
import logging
from agents.mcp import MCPServerStdio, MCPServerManager
from dotenv import load_dotenv
from contextlib import asynccontextmanager, AsyncExitStack

load_dotenv(override=True)

# 1. Base Configuration setup
logging.basicConfig(
    level=logging.INFO,                           # Set minimum severity level to log
    format='%(asctime)s - %(levelname)s - %(message)s', # Define log layout structure
    datefmt='%Y-%m-%d %H:%M:%S'                    # Style the timestamp output
)

# 2. Instantiate a module-level logger 
logger = logging.getLogger(__name__)

class StructuredResponse(BaseModel):
    message: str
    status: Literal["done", "pending", "failure"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    employee_server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "employee_mcp_server"],
            "cwd": ".",
        }
    )
    notification_server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "notification_mcp_server"],
            "cwd": ".",
        }
    )

    async with AsyncExitStack() as stack:
        employee_manager = await stack.enter_async_context(
            MCPServerManager([employee_server])
        )
        notify_manager = await stack.enter_async_context(
            MCPServerManager([notification_server])
        )

        app.state.employee_agent = Agent(
            name="employee_agent",
            model="gpt-4o-mini",
            instructions="""You're an employee agent who is responsible to perform employee specific operations such as:
                    - onboard_employee: Tool to use to insert the employee record into database.
                    - deboard_employee: Tool to use to delete the employee record from database.
                    - get_employee: Fetch the particular employee detail using email address.
                    - validate_employee_details: Validate the particular employee details using email address and provide the validated response.
                    Select the particular tool as per the user specific requirement.""",
            mcp_servers=employee_manager.active_servers,
            output_type=StructuredResponse,
            model_settings=ModelSettings(temperature=0.8),
        )

        app.state.notification_agent = Agent(
            name="notification_agent",
            model="gpt-4o-mini",
            instructions="""You're an notification agent who is responsible to perform notification specific operations such as:
                        - send_notification: Tool to send notification using recipient, sender and message.
                        - broadcast_notification: Tool to use send notification using recipient, sender(s) and message.
                        - schedule_notification: Tool to use send notification using recipient, sender(s) and message with cron expression.
                        Select the particular notification tool as per the user specific requirement.
                        
                        Given all required emails for notification purpose, Kindly use wherever needed:
                        - EMAIL: followup@myorgemail.com
                        REASON: Used for followup from all employees.
            
                        - EMAIL: hr@myorgemail.com
                        REASON: Used for HR specific purposes.
                        
                        - EMAIL: itsupport@myorgemail.com
                        REASON: Used for IT support specific purposes.""",
            mcp_servers=notify_manager.active_servers,
            output_type=StructuredResponse,
            model_settings=ModelSettings(temperature=0.8),
        )
        yield

    # async with MCPServerManager([employee_server]) as employee_manager:
    #     app.state.employee_agent = Agent(
    #         name="employee_agent",
    #         model="gpt-4o-mini",
    #         instructions="""
    #     You're an employee agent who is responsible to perform employee specific operations such as:
    #     - onboard_employee: Tool to use to insert the employee record into database.
    #     - deboard_employee: Tool to use to delete the employee record from database.
    #     - get_employee: Fetch the particular employee detail using email address.
    #     - validate_employee_details: Validate the particular employee details using email address and provide the validated response.
    #     Select the particular tool as per the user specific requirement.
    #     """,
    #         mcp_servers=employee_manager.active_servers,
    #         output_type=StructuredResponse,
    #         model_settings=ModelSettings(temperature=0.8)
    #     )
    #     async with MCPServerManager([notification_server]) as notify_manager:
    #         app.state.notification_agent = Agent(
    #             name="notification_agent",
    #             model="gpt-4o-mini",
    #             instructions="""
    #         You're an notification agent who is responsible to perform notification specific operations such as:
    #         - send_notification: Tool to send notification using recipient, sender and message.
    #         - broadcast_notification: Tool to use send notification using recipient, sender(s) and message.
    #         - schedule_notification: Tool to use send notification using recipient, sender(s) and message with cron expression.
    #         Select the particular notification tool as per the user specific requirement.
            
    #         Given all required emails for notification purpose, Kindly use wherever needed:
    #         - EMAIL: followup@myorgemail.com
    #         REASON: Used for followup from all employees.

    #         - EMAIL: hr@myorgemail.com
    #         REASON: Used for HR specific purposes.
            
    #         - EMAIL: itsupport@myorgemail.com
    #         REASON: Used for IT support specific purposes.
    #         """,
    #             mcp_servers=notify_manager.active_servers,
    #             output_type=StructuredResponse,
    #             model_settings=ModelSettings(temperature=0.8)
    #         )
    #         yield

app = FastAPI(lifespan=lifespan)

department_agent = Agent(
    name="department_agent",
    model="gpt-4o-mini",
    instructions="""
You're an department agent who is responsible to perform department specific operations such as:
- onboard_department: Tool to use to insert the department record into database.
- dismiss_department: Tool to use to delete the department record from database.
- get_departments: Tool to fetch all departments.
Select the particular department tool as per the user specific requirement.
""",
    output_type=StructuredResponse,
    model_settings=ModelSettings(temperature=0.5)
)



@app.post("/employee")
async def employee_agent(prompt: str):
    logger.info("Employee Agent starting...")
    response = await Runner.run(app.state.employee_agent, input=prompt, max_turns=5)
    logger.info(f"Employee Agent completed: {response.final_output}")
    return response.final_output
    
@app.post("/department")
async def department_agent(prompt: str):
    logger.info("Department Agent starting...")
    response = await Runner.run(department_agent, input=prompt, max_turns=5)
    logger.info(f"Department Agent completed: {response.final_output}")
    return response.final_output

@app.post("/notification")
async def notification_agent(prompt: str):
    logger.info("Notification Agent starting...")
    response = await Runner.run(app.state.notification_agent, input=prompt, max_turns=5)
    logger.info(f"Notification Agent completed: {response.final_output}")
    return response.final_output

def main():
    uvicorn.run(app, host="localhost", port=8011)

if __name__ == "__main__":
    main()