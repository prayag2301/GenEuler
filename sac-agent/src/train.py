# train.py
import os
import time
import torch
from torch_geometric.data import DataLoader
from config import DEVICE, SAC_CONFIG, TRAINING_CONFIG, DATA_PATHS, GNN_CONFIG
from utils import load_json_file, get_file_path, build_graph, normalize_features
from agent import SACAgent, ReplayBuffer
from db import initialize_db, save_training_result, get_latest_epoch

def load_training_data(iteration):
    """
    Load merged dynamic data and mesh data for a given iteration,
    then build and normalize the graph.
    """
    merged_path = os.path.join(DATA_PATHS["merged_data"], f"merged_data_{iteration}.json")
    mesh_path   = os.path.join(DATA_PATHS["mesh_data"], f"mesh_data_{iteration}.json")
    merged_data = load_json_file(merged_path)
    mesh_data   = load_json_file(mesh_path)
    graph = build_graph(merged_data, mesh_data, device=DEVICE)
    graph.x = normalize_features(graph.x)
    return graph, merged_data

def main():
    iteration = int(os.environ.get("ITERATION", 1))
    print(f"Loading training data for iteration {iteration}...")
    graph, merged_data = load_training_data(iteration)

    agent = SACAgent(
        input_dim=GNN_CONFIG["input_dim"],
        action_dim=3,  # add_offset is a 3‑vector per node
        sac_config=SAC_CONFIG,
        device=DEVICE
    )
    agent.model.to(DEVICE)
    print("✅ SAC Agent initialized.")

    replay_buffer = agent.replay_buffer
    # For demonstration, add one dummy transition.
    state = graph
    dummy_action = torch.zeros((graph.num_nodes, agent.action_dim), device=DEVICE)
    reward = torch.tensor(merged_data["individual_rewards"]["combined"], device=DEVICE)
    next_state = state
    done = torch.zeros(graph.num_nodes, dtype=torch.bool, device=DEVICE)
    for _ in range(1000):
        replay_buffer.push(state, dummy_action, reward, next_state, done)

    num_epochs = TRAINING_CONFIG["num_epochs"]
    log_interval = TRAINING_CONFIG["log_interval"]
    checkpoint_dir = TRAINING_CONFIG["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        batch = replay_buffer.sample()
        loss, actor_loss, critic_loss, entropy_loss = agent.train_step(batch)
        if epoch % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f}, Actor: {actor_loss:.4f}, "
                  f"Critic: {critic_loss:.4f}, Entropy: {entropy_loss:.4f} - {elapsed:.2f}s")
            checkpoint_path = os.path.join(checkpoint_dir, f"sac_checkpoint_epoch_{epoch}.pt")
            torch.save(agent.model.state_dict(), checkpoint_path)
            save_training_result(epoch, loss, checkpoint_path)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
