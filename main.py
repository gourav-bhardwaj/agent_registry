import os
from dotenv import load_dotenv
from fastapi import FastAPI
import numpy as np
from sentence_transformers import SentenceTransformer, util
from init import embed_agents_details, embed_organization_process
import uvicorn
from agents import Agent, Runner, ModelSettings
from pydantic import BaseModel
import logging
from task_spliter import generate_subtasks
from typing import Any
import requests
import json

# 1. Base Configuration setup
logging.basicConfig(
    level=logging.INFO,                           # Set minimum severity level to log
    format='%(asctime)s - %(levelname)s - %(message)s', # Define log layout structure
    datefmt='%Y-%m-%d %H:%M:%S'                    # Style the timestamp output
)

# 2. Instantiate a module-level logger 
logger = logging.getLogger(__name__)

class SelectedAgent(BaseModel):
    agent_name: str
    task: str
    reasoning: str

load_dotenv(override=True)

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = embed_agents_details(model)

agent_map = {doc.get("agent_name"): doc for doc in documents}

organization_processes = embed_organization_process(model)
top_k = len(documents)
agent_selector = Agent(
    name="agent_decider",
    model="gpt-4o-mini",
    instructions="""
You are an Agent Selector responsible for selecting the most appropriate
specialized agent to handle a user's request.

You will receive:
1. The user's query.
2. Metadata describing all available agents, including their:
   - agent_name
   - description
   - skill_set
   - tools

Your responsibilities:

- Analyze the user's query carefully.
- Compare the requested operation with the available agents' descriptions,
  skill sets, and tools.
- Select the single most appropriate agent.
- Do not perform the requested operation yourself.
- Do not call any tools.
- Select an agent only when its responsibilities are relevant to the user's query.
- Prefer the most specific agent rather than making assumptions.
- Return your selection using the required structured output format.

Examples:

User query:
"Create a new department called Engineering."

Selection:
department_agent

User query:
"Onboard John Smith as a new employee."

Selection:
employee_agent

User query:
"Send an email from john@company.com to alice@company.com."

Selection:
notification_agent

User query:
"Schedule a notification to alice@company.com for tomorrow at 10 AM."

Selection:
notification_agent

""",
    output_type=SelectedAgent,
    model_settings=ModelSettings(temperature=0.1)
)

@app.get("/agent")
async def find_semantic_search(query: str):
    query_embedding = model.encode(query)
    loaded_org_embeddings = np.load('organization_process_embed.npy')
    similarity_scores = util.cos_sim(query_embedding, loaded_org_embeddings)[0]
    top_index = np.argsort(-similarity_scores)[0]
    complete_task = await generate_subtasks("gpt-4o-mini", query, organization_processes[top_index])
    logger.info("sub_tasks type: %s", type(complete_task.tasks))

    # Now let's decide the agent for each sub task.
    loaded_embeddings = np.load('agents_embedding.npy')
    responses = []
    for sub_task in complete_task.tasks:
        task_desc = f"""
        Entity: {sub_task.entity}
        Task: {sub_task.task}
        Reason: {sub_task.reason}
        """
        st_embed = model.encode(task_desc)
        similarity_scores = util.cos_sim(st_embed, loaded_embeddings)[0]
        top_indices = np.argsort(-similarity_scores)[:top_k]
        #6. Print the matches
        logger.info(f"Query: '{sub_task.task}'\n")
        logger.info("Top Matches:")
        selected_agent_metadata = []
        for idx in top_indices:
            score = similarity_scores[idx].item()
            logger.info(f"- {documents[idx]} (Score: {score:.4f})")
            logger.info("-"*70)
            agent = documents[idx]
            selected_agent_metadata.append(agent)

        prompt = f"User Query:\n{sub_task.task}\n\nAvailable Agent Metadata:\n{selected_agent_metadata}"
        res = await Runner.run(agent_selector, input=prompt, max_turns=2)
        selected_agent = res.final_output
        logger.info("*"*70)

        agent_meta = agent_map.get(selected_agent.agent_name)
        if not agent_meta:
            logger.error("No metadata for selected agent %s", selected_agent.agent_name)
            continue

        logger.info(
            "\nAgent Name: %s\nAgent URL: %s\nSub-Task: %s\nEntity: %s\nReasoning: %s\n",
            selected_agent.agent_name,
            agent_meta,
            sub_task.task,
            sub_task.entity,
            sub_task.reason,
        )
        
        execution_prompt = f"""User Query: {query}
        Metadata Info:
        - Tools: {', '.join(agent_meta.get("tools", []))}
        - Sub-Task: {sub_task.task}
        - Entity: {sub_task.entity}
        - Reasoning: {sub_task.reason}
        """

        execution_result = call_agent(agent_meta.get("endpoint"), execution_prompt)
        responses.append(execution_result)
        logger.info(f"Output: {execution_result}")
        logger.info("*"*70)

    return responses

def call_agent(agent_url: str, prompt: str):
    # base_url = f"{agent_url}?prompt='{prompt}'"
    # response = requests.get(base_url)
    # if response.status_code == 200:
    #     return json.loads(response.content)
    # else:
    #     return None
    response = requests.post(
        agent_url,
        params={"prompt": prompt},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def main():
    uvicorn.run(app, host="localhost", port=8000)

if __name__ == "__main__":
    main()
