from typing import List
from pydantic import BaseModel, Field
from agents import Agent, Runner
from typing import Literal, Optional

class Task(BaseModel):
    order_id: int
    entity: Optional[str]
    task: str
    reason: Optional[str]


class CompleteTask(BaseModel):
    tasks: List[Task]

async def generate_subtasks(model: str, user_query: str, organization_process: str):
    if not organization_process:
        return CompleteTask(
            tasks=[
                Task(
                    order_id=1,
                    entity=None,
                    task=user_query,
                    reason="No matching organization process found"
                )
            ]
        )
    task_splitter_agent = Agent(
        name="task_splitter_agent",
        model=model,
        instructions="""
    You are the Task Splitter Agent in an organization management
    multi-agent system.

    Your responsibility is to analyze:
    1. The user's request.
    2. The organization's defined processes.

    You must convert the user's request into a sequence of small,
    ordered, executable tasks.

    Rules:

    1. Understand the user's intent before creating tasks.
    2. Identify the organization process that matches the user's request.
    3. Break the process into atomic tasks.
    4. Every task must represent one meaningful action.
    5. Tasks must be ordered according to their dependencies.
    6. Use the organization's process definition as the source of truth.
    7. Do not invent steps that are not required by the organization's process.
    8. Include notification, validation, approval, scheduling, and
    follow-up tasks when they are defined by the process.
    9. If one task depends on the result of another task, place the
    dependent task after it.
    10. The `entity` field should identify the primary business entity
        involved in the task, such as:
        - employee
        - department
        - notification
        - HR
        - IT
        - assignee_team
    11. The `task` field should contain a clear, actionable description.
    12. The `reason` field should explain why the task is required.
    13. Assign sequential integer values to `order_id`, starting from 1.
    14. Do not execute any task.
    15. Do not call any tools.
    16. Return only the structured CompleteTask response.

    Important:
    - Do not merge multiple independent actions into a single task.
    - Preserve the logical order of the organization process.
    - Do not skip prerequisite tasks.
    - If a notification is required after another operation, create it
    as a separate task.
    - If human approval is required, create a separate approval task.
    - If information is missing and the process requires a follow-up,
    create the follow-up notification task.

    Output format(json)
    - order_id: Order number must be unique
    - operation: ("write", "read", "delete", "update", "none") - Decide carefully based on the task context.
        such as: Capture/Fetch/Read/Retrive/Show - read
                 Create/Insert/Add/Put - write
                 Remove/Delete/Erase - delete
                 Change/Modify/Update - update
                 Others - None
    - entity: For whom we need to perform that task.
    - task: What we need to perform in a clear context.
    - reason: Why we need to perform this particular task.

    DON'T create unwanted steps which are really don't mention in ORGANIZATION PROCESSES. 
    """,
        output_type=CompleteTask,
    )

    # Run the agent and fetch the output.
    result = await Runner.run(
            task_splitter_agent,
            input=f"""
        USER QUERY:
        {user_query}

        ORGANIZATION PROCESSES:
        {organization_process}
        """,
        max_turns=5
        )

    return result.final_output