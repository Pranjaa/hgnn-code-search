import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
from inference import infer, code_map
from tqdm import tqdm  

ANNOTATED_FILE = r"data/annotatedData.jsonl"
TOP_K = 3
SIM_THRESHOLD = 0.7
MAX_THREADS = 8

query_groups = defaultdict(list)
with open(ANNOTATED_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        query_groups[item["query"]].append((item["function"], item["relevance"]))

embedder = SentenceTransformer("models/all-MiniLM-L6-v2")

def compute_similarity(code_a, code_b):
    emb1 = embedder.encode(code_a, convert_to_tensor=True)
    emb2 = embedder.encode(code_b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

def evaluate_query(query, gt_items, threshold=SIM_THRESHOLD, verbose=False):
    gt_funcs = [func for func, _ in gt_items]
    gt_rels = [rel for _, rel in gt_items]

    response = infer(query, k=TOP_K)
    predicted_ids = [idx for idx, _ in response]
    predicted_codes = [code_map.get(str(idx), "") for idx in predicted_ids]

    binary_relevance = []
    graded_relevance = []

    for pred_code in predicted_codes:
        similarities = [compute_similarity(pred_code, gt_code) for gt_code in gt_funcs]
        max_sim = max(similarities) if similarities else 0
        graded_relevance.append(max_sim)
        binary_relevance.append(int(max_sim >= threshold))

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

    for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating queries"):
        ndcg = future.result()
        ndcgs.append(ndcg)

print("\nFinal Evaluation Summary (Embedding Similarity):")
print(f"Mean NDCG:            {np.mean(ndcgs):.6f}")
