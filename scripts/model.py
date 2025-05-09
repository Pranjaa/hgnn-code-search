import torch
import torch.nn as nn

# --- Hypergraph Attention Layer ---
class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.4):
        super().__init__()
        self.node_attn = nn.MultiheadAttention(in_dim, num_heads, dropout=dropout, batch_first=True)
        self.edge_attn = nn.MultiheadAttention(in_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, 4 * out_dim),
            nn.GELU(),
            nn.Linear(4 * out_dim, out_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H, node_types):
        edge_indices = H._indices()
        edge_values = H._values()

        edge_repr = torch.zeros(H.size(1), X.size(1), device=X.device)
        edge_repr = edge_repr.scatter_add_(0, edge_indices[1].unsqueeze(-1).expand(-1, X.size(1)), X[edge_indices[0]] * edge_values.unsqueeze(-1))

        node_repr, _ = self.node_attn(X, edge_repr, edge_repr)
        node_repr = self.norm1(X + self.dropout(node_repr))

        attn_mask = (node_types.unsqueeze(1) == node_types.unsqueeze(0)).float()
        attn_output, _ = self.node_attn(node_repr, node_repr, node_repr, attn_mask=attn_mask)
        node_repr = self.norm2(node_repr + self.dropout(attn_output))

        return self.ffn(node_repr)

# --- Model ---
class AllSetTransformerModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, num_layers=3, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HypergraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, X, node_types, H):
        X = self.input_proj(X)
        for layer in self.layers:
            X = layer(X, H, node_types)
        return self.output_proj(X)

    def chunked_forward(self, combined_X, combined_types, H, chunk_size=512):
        all_embeddings = []
        num_nodes = combined_X.size(0)
        num_chunks = (num_nodes + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_nodes)

            chunk_X = combined_X[start:end]
            chunk_types = combined_types[start:end]

            type_to_id = {"function": 0, "concept": 1}  
            if isinstance(chunk_types, list):
                chunk_types = torch.tensor([type_to_id[t] for t in chunk_types], device=chunk_X.device)

            indices = H._indices()
            values = H._values()
            mask = (indices[0] >= start) & (indices[0] < end)
            filtered_indices = indices[:, mask].clone()
            filtered_indices[0] -= start
            filtered_values = values[mask]

            if filtered_indices.shape[1] == 0:
                filtered_indices = torch.zeros((2, 1), dtype=torch.long, device=H.device)
                filtered_values = torch.ones(1, device=H.device)

            H_chunk = torch.sparse_coo_tensor(
                filtered_indices, filtered_values,
                size=(end - start, H.shape[1])
            ).coalesce()

            X = self.input_proj(chunk_X)
            for j, layer in enumerate(self.layers):
                X = layer(X, H_chunk, chunk_types)

            if torch.isnan(X).any():
                X = torch.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)

            all_embeddings.append(X)

        return torch.cat(all_embeddings, dim=0)
    
    def encode(self, X, node_types, H):
        return self.forward(X, node_types, H)

# --- Hyperpath Contrastive Loss ---
class HyperpathContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, concept_embeddings, pos_fn_embeddings, neg_fn_embeddings, hyperpath_masks):
        B = concept_embeddings.size(0)
        hyperpaths = []
        for b in range(B):
            mask = hyperpath_masks[b].unsqueeze(-1)  
            selected = concept_embeddings[b] * mask
            hyperpath = selected.sum(dim=0) / (mask.sum() + 1e-6)
            hyperpaths.append(hyperpath)
        h_H = torch.stack(hyperpaths, dim=0)  

        pos_sim = self.cosine(h_H.unsqueeze(1), pos_fn_embeddings) / self.temperature
        neg_sim = self.cosine(h_H.unsqueeze(1), neg_fn_embeddings) / self.temperature

        pos_exp = torch.exp(pos_sim).sum(dim=1)
        neg_exp = torch.exp(neg_sim).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-6))
        return loss.mean()




