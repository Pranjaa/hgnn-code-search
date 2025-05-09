
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np

class HyperpathAttentionAggregator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_emb, concept_embs):  # concept_embs: (N, D)
        q = self.q_proj(query_emb)  # (1, D)
        attn_scores = torch.matmul(concept_embs, self.W(q).T)  # (N, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (N, 1)
        return (attn_weights * concept_embs).sum(dim=0, keepdim=True)  # (1, D)

def rank_hyperpaths_by_centrality(concept_to_edges):
    edge_freq = Counter()
    for edges in concept_to_edges.values():
        edge_freq.update(edges)
    sorted_edges = sorted(edge_freq.items(), key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in sorted_edges]

def expand_keywords(keywords, encoder, concept_embs, concept_texts, device, top_k=3):
    emb = encoder.encode(keywords)
    emb = torch.tensor(emb).mean(dim=0, keepdim=True).to(device)

    if isinstance(concept_embs, np.ndarray):
        concept_embs = torch.tensor(concept_embs, dtype=torch.float).to(device)

    sim = F.cosine_similarity(emb, concept_embs)
    top_indices = torch.topk(sim, k=top_k).indices
    new_terms = [concept_texts[i] for i in top_indices]
    return list(set(keywords + new_terms))
