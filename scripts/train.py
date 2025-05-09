import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import AllSetTransformerModel, HyperpathContrastiveLoss
from tqdm import tqdm
import random
import torch.nn.functional as F
from config import (EPOCHS, LR, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE, FEATURE_DIM, DEVICE, CHECKPOINT_DIR)

torch.autograd.set_detect_anomaly(True)

X = torch.load(r"intermediate/X.pt")
H = torch.load(r"intermediate/H.pt", weights_only=False)
node_types = torch.load(r"intermediate/node_types.pt")
function_to_concepts = torch.load(r"intermediate/function_to_concepts.pt")
nodes = torch.load(r"intermediate/nodes.pt")

print("Initializing model...")
model = AllSetTransformerModel(in_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = HyperpathContrastiveLoss(temperature=0.1)  
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def scipy_coo_to_torch_sparse(scipy_coo):
    coo = scipy_coo.tocoo() 
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()

def hyperpath_sampler(batch_fn_ids, function_to_concepts, H, node_types, num_neg=2, top_k=10):
    fn_to_concepts = []
    concept_counter = {}

    for fn_id in batch_fn_ids:
        concepts = function_to_concepts.get(fn_id, [])
        if not concepts:
            continue
        fn_to_concepts.append((fn_id, concepts))
        for concept_text in concepts:
            if concept_text in nodes:
                cid = nodes[concept_text]
                concept_counter[cid] = concept_counter.get(cid, 0) + 1

    sorted_concepts = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)
    concept_batch = [cid for cid, _ in sorted_concepts[:top_k]]
    concept_batch = [cid for cid in concept_batch if cid < H.shape[0] and node_types[cid] == "concept"]
    concept_batch_tensor = torch.tensor(concept_batch, dtype=torch.long, device=DEVICE)

    hyperpath_masks = []
    filtered_fn_ids = []

    for fn_id, concepts in fn_to_concepts:
        mask = torch.zeros(len(concept_batch_tensor), device=DEVICE)
        match = False
        concept_ids = {nodes[c] for c in concepts if c in nodes}

        for i, cid in enumerate(concept_batch_tensor.tolist()):
            if cid in concept_ids:
                mask[i] = 1
                match = True
        if match:
            filtered_fn_ids.append(fn_id)
            hyperpath_masks.append(mask)

    if not filtered_fn_ids:
        print("[DEBUG] No valid function-concept overlaps found in batch.")
        return None, None, None, None

    all_fn_ids = list(function_to_concepts.keys())
    neg_fn_ids = []
    for pos_id in filtered_fn_ids:
        candidates = [fid for fid in all_fn_ids if fid != pos_id]
        if len(candidates) < num_neg:
            sampled = candidates + random.choices(all_fn_ids, k=num_neg - len(candidates))
        else:
            sampled = random.sample(candidates, k=num_neg)
        neg_fn_ids.append(sampled)

    neg_fn_ids_tensor = torch.tensor([[int(fid) for fid in group] for group in neg_fn_ids], dtype=torch.long, device=DEVICE)

    return (
        concept_batch_tensor,
        torch.stack(hyperpath_masks),
        filtered_fn_ids,
        neg_fn_ids_tensor,
    )

def train(BATCH_SIZE=16, NUM_EPOCHS=3):
    print("Starting model training...")
    model.train()

    all_fn_ids = list(function_to_concepts.keys())
    fn_id_to_idx = {fn_id: idx for idx, fn_id in enumerate(sorted(set(all_fn_ids)))}

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        random.shuffle(all_fn_ids)
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        with tqdm(total=len(all_fn_ids), desc="Training Batches", unit="fn") as pbar:
            for i in range(0, len(all_fn_ids), BATCH_SIZE):
                batch_fn_ids = all_fn_ids[i:i + BATCH_SIZE]

                result = hyperpath_sampler(batch_fn_ids, function_to_concepts, H, node_types)

                if result is None or any(x is None for x in result):
                    pbar.update(len(batch_fn_ids))
                    continue

                concept_batch, hyperpath_masks, pos_fn_ids, neg_fn_ids = result

                if len(pos_fn_ids) == 0 or len(neg_fn_ids) == 0:
                    pbar.update(len(batch_fn_ids))
                    continue

                concept_X = X[concept_batch]
                concept_types = [node_types[cid] for cid in concept_batch.tolist()]
                H_torch = scipy_coo_to_torch_sparse(H).to(DEVICE)
                concept_embeddings = model.chunked_forward(concept_X, concept_types, H_torch, chunk_size=512)
                concept_embeddings = F.normalize(concept_embeddings, p=2, dim=-1)
                concept_embeddings = concept_embeddings.unsqueeze(0).repeat(len(pos_fn_ids), 1, 1)

                pos_fn_ids_idx = [fn_id_to_idx[fn] for fn in pos_fn_ids]
                pos_fn_ids_tensor = torch.tensor(pos_fn_ids_idx, dtype=torch.long, device=DEVICE)
                pos_X = X[pos_fn_ids_tensor]
                pos_types = [node_types[int(fn_id)] for fn_id in pos_fn_ids_tensor.tolist()]
                pos_fn_embeddings = model.chunked_forward(pos_X, pos_types, H_torch, chunk_size=512)
                pos_fn_embeddings = F.normalize(pos_fn_embeddings, p=2, dim=-1).unsqueeze(1)

                if isinstance(neg_fn_ids, list):
                    neg_fn_ids_tensor = torch.tensor(neg_fn_ids, dtype=torch.long, device=DEVICE)
                else:
                    neg_fn_ids_tensor = neg_fn_ids.to(DEVICE)

                flat_neg_ids = neg_fn_ids_tensor.view(-1)
                neg_X = X[flat_neg_ids]
                neg_types = [node_types[int(fid)] for fid in flat_neg_ids.tolist()]
                neg_fn_embeddings = model.chunked_forward(neg_X, neg_types, H_torch, chunk_size=512)
                neg_fn_embeddings = F.normalize(neg_fn_embeddings, p=2, dim=-1)
                neg_fn_embeddings = neg_fn_embeddings.view(len(pos_fn_ids), -1, pos_fn_embeddings.shape[-1])

                cos_sim_pos = F.cosine_similarity(concept_embeddings, pos_fn_embeddings, dim=-1).mean().item()
                neg_mean = neg_fn_embeddings.mean(dim=1)  
                neg_mean_expanded = neg_mean.unsqueeze(1).expand(-1, concept_embeddings.shape[1], -1)
                cos_sim_neg = F.cosine_similarity(concept_embeddings, neg_mean_expanded, dim=-1).mean().item()

                loss = loss_fn(concept_embeddings, pos_fn_embeddings, neg_fn_embeddings, hyperpath_masks)

                if torch.isnan(loss) or loss.item() == 0.0:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(len(batch_fn_ids))
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(all_fn_ids) // BATCH_SIZE)
        print(f"[INFO] Epoch {epoch + 1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.6f}")

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"allset_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at: {ckpt_path}")

    final_model_path = os.path.join(CHECKPOINT_DIR, "allset_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"[Final] Saved model at: {final_model_path}")

train()
