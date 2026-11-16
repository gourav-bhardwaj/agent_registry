import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

def embed_agents_details(model: SentenceTransformer):
    agents_json = Path("./a2a_agents_info.json").read_text(encoding="utf-8")
    agents = json.loads(agents_json)
    agent_details = []
    for agent in agents:
        agent_info = f"""
        Agent Name: {agent.get("agent_name")}
        Description: {agent.get("description")}
        """
        agent_details.append(agent_info)
    if not Path("agents_embedding.npy").exists():
        embeddings = model.encode(agent_details)
        np.save("agents_embedding.npy", embeddings)
    return agents

def embed_organization_process(model: SentenceTransformer):
    organization_process = Path("./organization_process.txt").read_text(encoding="utf-8")
    organization_processes = list(map(lambda x: x.strip(), organization_process.split("---")))
    if not Path("organization_process_embed.npy").exists():
        embeddings = model.encode(organization_processes)
        np.save("organization_process_embed.npy", embeddings)
    return organization_processes