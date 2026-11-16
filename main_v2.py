import asyncio
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
from pathlib import Path
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
import torch

# choose device (cuda -> mps -> cpu)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    "cpu"
)
model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
documents = embed_agents_details(model)
agent_map = {doc.get("agent_name"): doc for doc in documents}
organization_processes = embed_organization_process(model)

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
    # model_settings=ModelSettings(temperature=0.1)
)

# convert precomputed numpy embeddings to torch tensors on same device
ORG_EMB = torch.from_numpy(np.load("organization_process_embed.npy")).to(device)
AGENTS_EMB = torch.from_numpy(np.load("agents_embedding.npy")).to(device)
TOP_K = 3

def _tokenized_corpus():
    agents_info = Path("./a2a_agents_info.json")
    corpus = json.loads(agents_info.read_text(encoding="utf-8"))
    return [
        cp["description"].strip().lower().split()
        for cp in corpus
    ]

def get_bm25():
    from rank_bm25 import BM25Okapi
    corpus = _tokenized_corpus()
    return BM25Okapi(corpus=corpus)

bm25 = get_bm25()

def _top_k_metadata(embed, corpus_emb, documents, k=TOP_K):
    # `scores` is a torch tensor on the active device; use torch ops
    scores = util.cos_sim(embed, corpus_emb)[0]
    idxs = torch.argsort(scores, descending=True)[:k].cpu().tolist()
    return [documents[i] for i in idxs]

async def select_org_process(query: str):
    q_emb = model.encode(query, convert_to_tensor=True).to(device)
    scores = util.cos_sim(q_emb, ORG_EMB)[0]
    for i,score in enumerate(scores):
        print(f"{i} - {score}")
    idx = torch.argmax(scores).item()
    if scores[idx].item() < 0.4:
        return None
    return organization_processes[idx]

# async def select_agents_for_subtask(sub_task):
#     desc = f"Entity: {sub_task.entity}\nTask: {sub_task.task}\nReason: {sub_task.reason}"
#     emb = model.encode(desc, convert_to_tensor=True).to(device)
#     return _top_k_metadata(emb, AGENTS_EMB, documents)


async def select_agents_for_subtask(query: str):
    tokenized_query = query.lower().split()
    bm25_scores = torch.tensor(bm25.get_scores(tokenized_query), dtype=torch.float32).to(device)
    emb = model.encode(query, convert_to_tensor=True).to(device)
    semantic_scores = util.cos_sim(emb, AGENTS_EMB)[0]

    # normalize both to 0..1 so one score type does not dominate the other
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)

    # weighted mix
    combined = 0.5 * semantic_norm + 0.5 * bm25_norm

    topk = torch.topk(combined, k=3).indices.cpu().tolist()
    return [documents[i]["description"] for i in topk]

@app.get("/agent")
async def find_semantic_search(query: str):
    # org = await select_org_process(query)
    # complete_task = await generate_subtasks("gpt-4o-mini", query, org)
    #async def handle(st):
    metas = await select_agents_for_subtask(query)
    prompt = f"User Query:\n{query}\n\nAvailable Agent Metadata:\n{metas}"
    res = await Runner.run(agent_selector, input=prompt)
    selected_agent = res.final_output
    agent_meta = agent_map.get(selected_agent.agent_name)
    if not agent_meta:
        logger.error("No metadata for selected agent %s", selected_agent.agent_name)
        return {
            "agent_url": None,
            "task": query
        }
    return {
        "agent_url": agent_meta["endpoint"],
        "task": query
    }
    # return await asyncio.gather(*(handle(st) for st in complete_task.tasks))

def main():
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()