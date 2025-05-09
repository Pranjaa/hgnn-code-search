import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score
from transformers import RobertaTokenizer, RobertaModel
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

tokenizer = RobertaTokenizer.from_pretrained("models/codebert-base")
model = RobertaModel.from_pretrained("models/codebert-base")
model.eval()

ANNOTATED_FILE = "data/annotatedData.jsonl"
TOP_K = 3
MAX_THREADS = 8
SIM_THRESHOLD = 0.4

query_groups = defaultdict(list)
with open(ANNOTATED_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        query_groups[item["query"]].append((item["function"], item["relevance"]))

def compute_similarity(code_a, code_b):
    inputs = tokenizer([code_a, code_b], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  
    emb1, emb2 = embeddings[0], embeddings[1]
    sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
    return sim.item()

def evaluate_query(query, gt_items):
    gt_funcs = [func for func, _ in gt_items]
    gt_rels = [rel for _, rel in gt_items]

    predicted_codes = gt_funcs[:TOP_K]

    binary_relevance = []
    graded_relevance = []

    for pred_code in predicted_codes:
        similarities = [compute_similarity(pred_code, gt_code) for gt_code in gt_funcs]
        max_sim = max(similarities) if similarities else 0
        graded_relevance.append(max_sim)
        binary_relevance.append(int(max_sim >= SIM_THRESHOLD))

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

print("\nEvaluation Metrics Summary for CodeBERT:")
print(f"Mean NDCG:            {np.mean(ndcgs):.6f}")
