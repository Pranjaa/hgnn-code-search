import torch
import torch.nn.functional as F
import re
from collections import defaultdict
from config import DEVICE, TOP_K, TOP_K_CONCEPTS
from builder import E5Encoder
from modules import HyperpathAttentionAggregator, rank_hyperpaths_by_centrality
import numpy as np

torch.cuda.empty_cache()

encoder = E5Encoder()

X = torch.load("intermediate/X.pt")
H = torch.load("intermediate/H.pt", weights_only=False)
node_types = torch.load("intermediate/node_types.pt")
function_to_concepts = torch.load("intermediate/function_to_concepts.pt")
concept_embs = torch.load("intermediate/concept_embs.pt", weights_only=False)
node_embs = torch.load("intermediate/node_embs.pt", weights_only=False)
concept_indices = torch.load("intermediate/concept_indices.pt")
function_indices = torch.load("intermediate/function_indices.pt")
code_map = torch.load("intermediate/code_map.pt")
stopwords = set(torch.load("intermediate/stopwords.pt"))  
nodes = torch.load("intermediate/nodes.pt")

id_to_node = {v: k for k, v in nodes.items()}
concept_texts = [id_to_node[i] for i in concept_indices]
function_texts = [id_to_node[i] for i in function_indices]

if not isinstance(concept_embs, torch.Tensor):
    concept_embs = torch.tensor(np.array(concept_embs), dtype=torch.float, device=DEVICE)
else:
    concept_embs = concept_embs.to(DEVICE)

if not isinstance(node_embs, torch.Tensor):
    node_embs = torch.tensor(np.array(node_embs), dtype=torch.float, device=DEVICE)
else:
    node_embs = node_embs.to(DEVICE)

aggregator = HyperpathAttentionAggregator(embed_dim=node_embs.shape[1]).to(DEVICE)

def scipy_coo_to_torch_sparse(scipy_coo):
    coo = scipy_coo.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()

def extract_keywords(text):
    return [w for w in re.findall(r'\b\w+\b', text.lower()) if w not in stopwords]

def build_edge_maps(H_torch):
    indices = H_torch.indices()
    node_indices, edge_indices = indices[0].tolist(), indices[1].tolist()

    edge_to_nodes = defaultdict(set)
    concept_to_edges = defaultdict(set)

    for n_id, e_id in zip(node_indices, edge_indices):
        edge_to_nodes[e_id].add(n_id)
        if node_types[n_id] == "concept":
            concept_to_edges[n_id].add(e_id)

    return edge_to_nodes, concept_to_edges

H_torch = scipy_coo_to_torch_sparse(H).to(DEVICE)
edge_to_nodes, concept_to_edges = build_edge_maps(H_torch)

def get_hyperpaths(top_concepts):
    ranked_edges = rank_hyperpaths_by_centrality(concept_to_edges)
    valid_edges = set.union(*(concept_to_edges.get(c_id, set()) for c_id in top_concepts))
    hyperpaths = []

    for e_id in ranked_edges:
        if e_id not in valid_edges:
            continue
        nodes_in_edge = list(edge_to_nodes.get(e_id, []))
        concept_ids = [n_id for n_id in nodes_in_edge if node_types[n_id] == "concept"]
        if concept_ids:
            hyperpaths.append((e_id, concept_ids))
        if len(hyperpaths) >= TOP_K_CONCEPTS:
            break

    return hyperpaths

@torch.no_grad()
def infer(query_text, k=TOP_K):
    query_emb_np = encoder.encode(query_text)  
    query_emb = torch.tensor(query_emb_np, dtype=torch.float, device=DEVICE).unsqueeze(0)
    query_emb = F.normalize(query_emb, dim=1)

    concept_embs_norm = F.normalize(concept_embs, dim=1)
    sim_scores = F.cosine_similarity(query_emb, concept_embs_norm)
    top_concepts = torch.topk(sim_scores, k=TOP_K_CONCEPTS).indices.tolist()
    top_concept_ids = [concept_indices[i] for i in top_concepts]

    hyperpaths = get_hyperpaths(top_concept_ids)
    function_embs_tensor = node_embs[function_indices]

    scores = torch.full((len(function_indices),), float('-inf'), device=DEVICE)

    for _, concept_ids in hyperpaths:
        h_concept_embs = node_embs[concept_ids]
        h_H = aggregator(query_emb, h_concept_embs)

        mean_sim = torch.mean(F.cosine_similarity(query_emb, h_concept_embs))
        if mean_sim.item() < 0.3:
            continue

        cosine_sim = F.cosine_similarity(h_H, function_embs_tensor)

        concept_set = set(concept_ids)
        overlaps = torch.tensor([
            len(set(function_to_concepts.get(f_id, [])) & concept_set) / max(len(concept_set), 1)
            for f_id in function_indices
        ], dtype=torch.float, device=DEVICE)

        key_bon = torch.tensor([
            sum(kw.lower() in function_texts[i].lower() for kw in extract_keywords(query_text))
            for i in range(len(function_indices))
        ], dtype=torch.float, device=DEVICE)

        new_scores = 0.7 * cosine_sim + 0.2 * overlaps + 0.1 * key_bon
        scores = torch.max(scores, new_scores)

    topk_indices = torch.topk(scores, k).indices
    topk_ids = [function_indices[i] for i in topk_indices]

    rank_query_emb = torch.tensor(encoder.encode(query_text), dtype=torch.float)
    rank_query_emb = F.normalize(rank_query_emb, dim=0)

    seen_ids = set()
    ranked = []
    for idx in topk_ids:
        if idx in seen_ids:
            continue
        seen_ids.add(idx)
        func_text = function_texts[function_indices.index(idx)]
        func_emb = torch.tensor(encoder.encode(func_text), dtype=torch.float)
        func_emb = F.normalize(func_emb, dim=0)
        score = F.cosine_similarity(rank_query_emb, func_emb, dim=0).item()
        ranked.append((idx, score))

    ranked.sort(key=lambda x: -x[1])
    return ranked


if __name__ == "__main__":
    query = ""
    results = infer(query, k=TOP_K)

    print("\nTop Matching Functions:")
    topk_ids = []
    for idx, score in results:
        func_text = id_to_node.get(idx, "[Unknown Function]")
        code = code_map[str(idx)]
        print(f"Function ID: {idx}, \nScore: {score:.4f}, \nText: {func_text}..., \n Code: {code}")
        print("\n------------------------------------------")
        topk_ids.append(idx)




