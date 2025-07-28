# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from model import SACGNNModel  # Our GNN-based SAC model defined in model.py
from config import DEVICE, SAC_CONFIG, GNN_CONFIG
from torch.optim import AdamW
from collections import deque

# -------------------- Replay Buffer --------------------
class ReplayBuffer:
    def __init__(self, capacity, batch_size, device=DEVICE):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """
        Each transition is a tuple:
          (state, action, reward, next_state, done)
        where state and next_state are PyTorch Geometric Data objects.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly sample a batch of transitions and collate them.
        """
        batch = random.sample(self.buffer, self.batch_size)
        from torch_geometric.data import Batch
        states, actions, rewards, next_states, dones = zip(*batch)
        # build PyG batches
        states_batch = Batch.from_data_list(states).to(self.device)
        next_states_batch = Batch.from_data_list(next_states).to(self.device)
        # stack & flatten actions: (B, N, A) → (B*N, A)
        actions = torch.stack(actions).to(self.device)
        B, N, A = actions.shape
        actions = actions.view(B * N, A)
        # replicate scalar rewards & dones per node: (B,1) → (B,1) → (B*N,1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        rewards = rewards.repeat_interleave(N, dim=0)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = dones.repeat_interleave(N, dim=0)
        return states_batch, actions, rewards, next_states_batch, dones

    def __len__(self):
        return len(self.buffer)

# -------------------- SACAgent --------------------
class SACAgent:
    def __init__(self, input_dim, action_dim, sac_config, device=DEVICE):
        self.device = device
        self.action_dim = action_dim

        # Actor (policy) network
        self.actor = SACGNNModel(GNN_CONFIG, action_dim, role="actor").to(self.device)
        # Two critic networks
        self.critic1 = SACGNNModel(GNN_CONFIG, action_dim, role="critic").to(self.device)
        self.critic2 = SACGNNModel(GNN_CONFIG, action_dim, role="critic").to(self.device)
        # Target critics
        self.target_critic1 = SACGNNModel(GNN_CONFIG, action_dim, role="critic").to(self.device)
        self.target_critic2 = SACGNNModel(GNN_CONFIG, action_dim, role="critic").to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer  = AdamW(self.actor.parameters(),
                                      lr=sac_config["actor_lr"],
                                      weight_decay=1e-4)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = AdamW(critic_params,
                                       lr=sac_config["critic_lr"],
                                       weight_decay=1e-4)
        self.alpha = sac_config["entropy_coeff"]
        self.discount_factor = sac_config["discount_factor"]
        self.tau = sac_config["tau"]

        self.replay_buffer = ReplayBuffer(sac_config["replay_buffer_size"],
                                          sac_config["batch_size"],
                                          device=self.device)
        
        # track recent raw losses to set a dynamic cap
        self.loss_history = deque(maxlen=10)

    def get_action(self, data):
        """
        Given a state (graph data), compute per-node outputs using the actor network.
        Returns:
          - add_offset: a (num_nodes, 3) tensor; each row is the proposed offset to add.
          - removal_score: a (num_nodes, 1) tensor; each row is the removal score.
        """
        self.actor.eval()
        with torch.no_grad():
            add_offset, removal_score = self.actor(data)
        self.actor.train()
        return add_offset, removal_score

    def update_target_networks(self):
        """
        Soft-update the target critic networks.
        target_param = tau * param + (1 - tau) * target_param
        """
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_step(self, batch):
        """
        Performs one SAC update:
          - samples batch
          - updates critics, actor
          - soft‑updates target networks
        Returns total_loss for logging.
        """
        states, actions, rewards, next_states, dones = batch

        # 1) Critic loss: clamp TD‐targets + Huber loss
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.discount_factor * q_next
            target_q = target_q.clamp(-5.0, 5.0)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        huber = nn.SmoothL1Loss()
        critic_loss = huber(q1, target_q) + huber(q2, target_q)

        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 2) Actor loss (unchanged)
        new_actions, log_probs = self.actor.sample(states)
        q1_pi = self.critic1(states, new_actions)
        q2_pi = self.critic2(states, new_actions)
        q_pi  = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - q_pi).mean()

        # Actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 3) Soft updates
        self.update_target_networks()

        # raw total (could be very large or NaN)
        raw_critic = critic_loss.item()
        raw_actor  = actor_loss.item()
        raw_total  = raw_critic + raw_actor

        # 1) record it
        self.loss_history.append(raw_total)

        # 2) compute a high‐percentile cap (99th pct of recent history)
        import numpy as np
        if len(self.loss_history) >= 2:
            L_max = float(np.percentile(list(self.loss_history), 99))
        else:
            L_max = 10.0   # fallback for the very first steps
        # avoid division by ~zero
        L_max = max(L_max, 1e-6)

        # 3) scale into [0,100]
        loss_pct = min(raw_total / L_max * 100.0, 100.0)
 
        return loss_pct, raw_actor, raw_critic, log_probs.mean().item()
