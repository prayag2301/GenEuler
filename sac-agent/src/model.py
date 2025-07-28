# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SharedGNN(nn.Module):
    """
    Shared GNN layers that process node features.
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SharedGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

class BranchNetwork(nn.Module):
    """
    A branch MLP of configurable depth and width.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BranchNetwork, self).__init__()
        self.layers = nn.ModuleList()
        # first layer: input_dim → hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # middle hidden layers (hidden_dim → hidden_dim)
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for lin in self.layers:
            x = F.relu(lin(x))
        return self.out(x)

class FusionLayer(nn.Module):
    """
    Fusion layer that combines outputs from the shared GNN and the branches.
    """
    def __init__(self, input_dim, fusion_dim):
        super(FusionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, fusion_dim)
    
    def forward(self, x):
        return F.relu(self.fc(x))

class MetaController(nn.Module):
    """
    Meta controller that produces the final node-level adjustment action.
    In this example, we output one continuous scalar per node.
    """
    def __init__(self, input_dim, action_dim=1):
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, action_dim)
        self.scale = 1.0      # was hard‑coded 5.0
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.scale * torch.tanh(self.fc2(x))

class SACGNNModel(nn.Module):
    """
    The full SAC-GNN model:
      - Shared GNN layers.
      - Two branches (for FEM and Topology).
      - A fusion layer that concatenates (shared_out, fem_out, topo_out).
      - A meta controller head that produces (action, log_prob).
    'role' is an optional keyword (e.g., 'actor' or 'critic') to match agent code.
    """
    def __init__(self, gnn_config, action_dim=1, role=None):
        super().__init__()
        self.shared_gnn = SharedGNN(
            input_dim   = gnn_config["input_dim"],
            hidden_dim  = gnn_config["shared_hidden_dim"],
            num_layers  = gnn_config["shared_num_layers"],
        )

        # each branch gets shared_out + its per‐node reward
        branch_in_dim = gnn_config["shared_hidden_dim"] + 1
        self.fem_branch = BranchNetwork(
            input_dim  = branch_in_dim,
            hidden_dim = gnn_config["branch_hidden_dim"],
            output_dim = gnn_config["output_dim"],
            num_layers = gnn_config["branch_num_layers"],
        )
        self.topo_branch = BranchNetwork(
            input_dim  = branch_in_dim,
            hidden_dim = gnn_config["branch_hidden_dim"],
            output_dim = gnn_config["output_dim"],
            num_layers = gnn_config["branch_num_layers"],
        )

        fusion_dim = gnn_config["fusion_dim"]
        
        self.fusion = FusionLayer(gnn_config["shared_hidden_dim"] + 2 * gnn_config["output_dim"], fusion_dim)
        
        # actor heads
        self.meta_controller_remove = MetaController(fusion_dim, 1)
        self.meta_controller_add    = MetaController(fusion_dim, 3)
        # critic head: takes fusion_out + action → Q-value
        self.critic_head = nn.Linear(fusion_dim + action_dim, 1)
    
    def forward(self, data, action=None):
        """
        Returns: (action, log_prob).
        For real usage, you'd compute a distribution (e.g. TanhNormal) and sample from it.
        Here we produce a dummy log_prob = 0 for each node.
        """
        x, edge_index = data.x, data.edge_index
        shared_out = self.shared_gnn(x, edge_index)        # shape: (num_nodes, hidden_dim)
        fem_in  = torch.cat([shared_out, data.fem_reward], dim=1)
        topo_in = torch.cat([shared_out, data.topology_reward], dim=1)
        fem_out  = self.fem_branch(fem_in)   # shape: (num_nodes, output_dim)
        topo_out = self.topo_branch(topo_in)
        
        fusion_input = torch.cat([shared_out, fem_out, topo_out], dim=1)
        fusion_out = self.fusion(fusion_input)             # shape: (num_nodes, fusion_dim)
        
        if action is None:
            # actor mode: return offsets & scores
            removal_score   = self.meta_controller_remove(fusion_out)
            addition_offset = self.meta_controller_add(fusion_out)
            return addition_offset, removal_score
        else:
            # critic mode: compute Q(s,a)
            # action: (num_nodes, action_dim)
            cat = torch.cat([fusion_out, action], dim=1)
            q  = self.critic_head(cat)  # (num_nodes,1)
            return q

    def sample(self, data):
        """
        For compatibility with train_step:
          - runs a forward pass
          - returns (action, log_prob)
        Here we use a dummy zero log‐prob.
        """
        addition_offset, removal_score = self.forward(data)
        # log_prob shape must broadcast to removal_score
        zero_logp = torch.zeros_like(removal_score, device=removal_score.device)
        return addition_offset, zero_logp
