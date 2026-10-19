import torch
import torch.nn as nn
import numpy as np

class TimeEncoderV2(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Learnable frequencies and phase shifts for continuous time
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float().reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        tt = t.clone().detach() if not t.requires_grad else t
        return torch.cos(self.w(tt))

class NodeMemoryV2(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(num_nodes))

    def get_memory(self, node_ids, current_t=None):
        # Detach to prevent BPTT across entire history
        mem = self.memory[node_ids].detach()
        last_t = self.last_update[node_ids].detach()
        
        # [V2 Robustness]: Memory Staleness Decay
        # If the account has been dormant, decay its memory before processing
        if current_t is not None:
            delta_t = current_t - last_t
            # Exponential decay factor (lambda = 0.001)
            decay_factor = torch.exp(-0.001 * delta_t).unsqueeze(-1)
            mem = mem * decay_factor
            
        return mem, last_t

    def set_memory(self, node_ids, values, timestamps):
        self.memory[node_ids] = values
        self.last_update[node_ids] = timestamps

    def reset_memory(self):
        self.memory.zero_()
        self.last_update.zero_()

class TemporalEncoderV2(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int, edge_dim: int, time_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        self.time_encoder = TimeEncoderV2(time_dim)
        self.memory = NodeMemoryV2(num_nodes, node_dim)
        
        # Message input: message features + time features + neighbor memory
        self.message_dim = edge_dim + time_dim + node_dim
        self.updater = nn.GRUCell(self.message_dim, node_dim)
        
        # [V2 Robustness]: Structural Neighborhood Attention Combiner
        self.attention_agg = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Tanh(),
            nn.Linear(node_dim, node_dim)
        )
        
    def forward(self, src, dst, t, msg):
        """
        Process a batch of continuous-time events using strictly safe chronological updates.
        """
        batch_size = src.shape[0]
        
        # [V2 Robustness]: Intra-Batch Chronological Sorting
        # Resolves high-frequency duplicate node collisions in the same batch
        sort_idx = torch.argsort(t)
        src_s, dst_s, t_s, msg_s = src[sort_idx], dst[sort_idx], t[sort_idx], msg[sort_idx]
        
        out_src_mem = []
        out_dst_mem = []
        
        # Sequential iteration prevents stale writes for rapid mule transactions
        for i in range(batch_size):
            u = src_s[i:i+1]
            v = dst_s[i:i+1]
            ts = t_s[i:i+1]
            m = msg_s[i:i+1]
            
            u_mem, u_last_t = self.memory.get_memory(u, ts)
            v_mem, v_last_t = self.memory.get_memory(v, ts)
            
            u_time_delta = self.time_encoder(ts - u_last_t)
            v_time_delta = self.time_encoder(ts - v_last_t)
            
            msg_to_u = torch.cat([m, u_time_delta, v_mem], dim=-1)
            msg_to_v = torch.cat([m, v_time_delta, u_mem], dim=-1)
            
            new_u_mem = self.updater(msg_to_u, u_mem)
            new_v_mem = self.updater(msg_to_v, v_mem)
            
            self.memory.set_memory(u, new_u_mem, ts)
            self.memory.set_memory(v, new_v_mem, ts)
            
            out_src_mem.append(new_u_mem)
            out_dst_mem.append(new_v_mem)
            
        updated_src = torch.cat(out_src_mem, dim=0)
        updated_dst = torch.cat(out_dst_mem, dim=0)
        
        # Revert sort to match original graph indices for Loss computation
        inv_sort_idx = torch.argsort(sort_idx)
        updated_src = updated_src[inv_sort_idx]
        updated_dst = updated_dst[inv_sort_idx]
        
        # [V2 Robustness]: Structural Neighborhood Attention
        # Aggregate structural context dynamically rather than isolating to pairs
        src_context = self.attention_agg(torch.cat([updated_src, updated_dst], dim=-1))
        dst_context = self.attention_agg(torch.cat([updated_dst, updated_src], dim=-1))
        
        return src_context, dst_context
