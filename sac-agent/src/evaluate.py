import os
import torch
from config import DEVICE, GNN_CONFIG, DATA_PATHS, SAC_CONFIG
from utils import load_json_file, build_graph, normalize_features
from agent import SACAgent
from torch_geometric.data import Data

def evaluate_latest():
    iteration = int(os.environ.get("ITERATION", 1))
    merged_path = os.path.join(DATA_PATHS["merged_data"], f"merged_data_{iteration}.json")
    mesh_path = os.path.join(DATA_PATHS["mesh_data"], f"mesh_data_{iteration}.json")
    merged_data = load_json_file(merged_path)
    mesh_data = load_json_file(mesh_path)
    graph = build_graph(merged_data, mesh_data, device=DEVICE)
    graph.x = normalize_features(graph.x)
    
    action_dim = 3
    agent = SACAgent(input_dim=GNN_CONFIG["input_dim"], action_dim=action_dim,
                     sac_config=SAC_CONFIG, gnn_config=GNN_CONFIG, device=DEVICE)
    checkpoint_path = os.path.join("./checkpoints", "sac_checkpoint_epoch_latest.pt")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    agent.actor.eval()  # Put actor in evaluation mode.
    
    with torch.no_grad():
        action = agent.get_action(graph)
    print("Action:", action.cpu().numpy())

if __name__ == "__main__":
    evaluate_latest()
