import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

model = SentenceTransformer("models/e5-small-v2")

ANNOTATED_FILE = "data/annotatedData.jsonl"
TOP_K = 3
MAX_THREADS = 8
SIM_THRESHOLD = 0.7

query_groups = defaultdict(list)
with open(ANNOTATED_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        query_groups[item["query"]].append((item["function"], item["relevance"]))

def evaluate_query(query, gt_items):
    gt_funcs = [func for func, _ in gt_items]
    gt_rels = [rel for _, rel in gt_items]

    query_emb = model.encode("query: " + query, convert_to_tensor=True)
    func_embs = model.encode(["passage: " + f for f in gt_funcs], convert_to_tensor=True)

    sims = util.cos_sim(query_emb, func_embs)[0]
    top_indices = sims.topk(k=TOP_K).indices.tolist()

    binary_relevance = []
    graded_relevance = []

    for i in range(TOP_K):
        idx = top_indices[i]
        sim = sims[idx].item()
        graded_relevance.append(sim)
        binary_relevance.append(int(sim >= SIM_THRESHOLD))

    ideal_rels = gt_rels[:TOP_K] + [0] * (TOP_K - len(gt_rels))
    padded_pred = graded_relevance + [0] * (TOP_K - len(graded_relevance))
    ndcg = ndcg_score([ideal_rels], [padded_pred])

    return ndcg

ndcgs = []
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = {
        executor.submit(evaluate_query, query, gt_items): query
        for query, gt_items in query_groups.items()
    }
    for future in as_completed(futures):
        ndcg = future.result()
        ndcgs.append(ndcg)

print("\nEvaluation Metrics Summary for E5-small-v2:")
print(f"Mean NDCG:            {np.mean(ndcgs):.6f}")
